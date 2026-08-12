"""
llm.py
Central LLM client shared by every agent node in the graph (Clarifier,
Planner, Synthesiser, Verifier, Finaliser, Refiner). Researcher does not
need one -- it only calls src/retrieval.py.

One place to configure model/provider/temperature so the nodes stay
consistent and swapping providers later doesn't mean touching every file.
"""

import os

from dotenv import load_dotenv
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
