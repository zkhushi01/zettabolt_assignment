"""
llm.py
Central LLM client shared by every agent node in the graph (Clarifier,
Planner, Researcher [only on a retry, to reformulate a failed query],
Synthesiser, Verifier, Finaliser).

One place to configure model/provider/temperature so the nodes stay
consistent and swapping providers later doesn't mean touching every file.

Also owns per-run LLM usage tracking (call count + tokens) for the eval
harness's cost metric. Implemented via the same context-var registration
LangChain's own get_openai_callback() uses internally (register_configure_hook),
NOT by passing config={"callbacks": [...]} at each node's .invoke() call --
that approach was tried first and silently failed to fire through
.with_structured_output() (it builds a fresh internal runnable that doesn't
inherit callbacks bound via .with_config() on the base model). The
context-var route attaches to every LLM call transparently, provider- and
node-file-agnostic, with zero changes needed in clarifier.py/planner.py/etc.
"""

import os
from contextvars import ContextVar

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---- Config ----
# "gemini-flash-latest" is Google's alias for their current recommended
# Flash model -- pinned version names (e.g. "gemini-2.5-flash") get cut off

MODEL_NAME = "gemini-flash-lite-latest"

DEFAULT_TEMPERATURE = 0.0


class LLMConfigError(Exception):
    """Raised when the Gemini API key is missing."""


_llm_cache: dict[float, ChatGoogleGenerativeAI] = {}


def get_llm(temperature: float = DEFAULT_TEMPERATURE) -> ChatGoogleGenerativeAI:
    """
    Returns a cached Gemini client for the given temperature. Nodes that
    need different behavior (e.g. a slightly higher temperature for
    Refiner's rewriting) pass their own value; everything else should use
    the deterministic default.
    """
    if temperature not in _llm_cache:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMConfigError(
                "GOOGLE_API_KEY is not set. Add it to a local .env file "
                "(see .env.example) -- get a free key at "
                "https://aistudio.google.com/apikey"
            )
        _llm_cache[temperature] = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=temperature,
            google_api_key=api_key,
        )
    return _llm_cache[temperature]


# ---- Usage tracking (eval harness cost metric) ----

class _UsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        for generation_list in response.generations:
            for generation in generation_list:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)


_usage_tracker_var: ContextVar = ContextVar("usage_tracker_var", default=None)
register_configure_hook(_usage_tracker_var, True)


def reset_usage_tracker() -> None:
    """Call before a graph run to start counting LLM calls/tokens from zero."""
    _usage_tracker_var.set(_UsageTracker())


def get_usage_snapshot() -> dict:
    """Returns {llm_calls, input_tokens, output_tokens, total_tokens} accumulated
    since the last reset_usage_tracker() call. All zero if tracking was never
    started (e.g. normal CLI/API use outside the eval harness)."""
    tracker = _usage_tracker_var.get()
    if tracker is None:
        return {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "llm_calls": tracker.calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
    }
