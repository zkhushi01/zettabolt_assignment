import { useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'

mermaid.initialize({ startOnLoad: false, theme: 'neutral' })

// Node ids as declared in src/graph.py's graph.add_node(...) calls -- kept
// here only to know which mermaid node ids to highlight as "done", not to
// duplicate the graph's actual wiring (that comes from /api/topology).
const GRAPH_NODE_IDS = ['clarifier', 'planner', 'researcher', 'synthesiser']

function TopologyDiagram({ baseMermaid, doneNodes }) {
  const containerRef = useRef(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!baseMermaid) return
    const highlighted = GRAPH_NODE_IDS.filter((id) => doneNodes.has(id))
    let source = baseMermaid
    if (highlighted.length) {
      source += `\nclassDef done fill:#22c55e,color:#fff,stroke:#16a34a;\nclass ${highlighted.join(',')} done;`
    }
    let cancelled = false
    mermaid
      .render(`topology-${Date.now()}`, source)
      .then(({ svg }) => {
        if (!cancelled && containerRef.current) containerRef.current.innerHTML = svg
      })
      .catch((err) => !cancelled && setError(err.message))
    return () => {
      cancelled = true
    }
  }, [baseMermaid, doneNodes])

  if (error) return <div className="text-sm text-red-600 dark:text-red-400">Diagram failed to render: {error}</div>
  return <div ref={containerRef} className="overflow-x-auto" />
}

function Badge({ children, tone = 'neutral' }) {
  const tones = {
    neutral: 'bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300',
    green: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300',
    amber: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
    purple: 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
  }
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>
}

function ClarifierCard({ update }) {
  return (
    <div className="space-y-2">
      {update.clarified_question && (
        <p className="text-sm">
          <span className="font-medium">Clarified question:</span> {update.clarified_question}
        </p>
      )}
      {update.assumptions?.length > 0 && (
        <div className="text-sm">
          <span className="font-medium">Assumptions recorded:</span>
          <ul className="mt-1 list-inside list-disc text-neutral-600 dark:text-neutral-400">
            {update.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      )}
      {update.clarifying_question_pending && (
        <p className="text-sm text-amber-700 dark:text-amber-400">
          Pending note for Clarifier: {update.clarifying_question_pending}
        </p>
      )}
    </div>
  )
}

function PlannerCard({ update }) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Badge tone={update.retrieval_needed ? 'green' : 'neutral'}>
          retrieval_needed: {String(update.retrieval_needed)}
        </Badge>
        <Badge tone={update.still_ambiguous ? 'amber' : 'neutral'}>
          still_ambiguous: {String(update.still_ambiguous)}
        </Badge>
      </div>
      {update.sub_questions?.length > 0 && (
        <ul className="space-y-1 text-sm">
          {update.sub_questions.map((sq) => (
            <li key={sq.id} className="rounded border border-neutral-200 px-2 py-1 dark:border-neutral-800">
              <span className="mr-2 font-mono text-xs text-neutral-400">{sq.id}</span>
              {sq.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function EvidenceRow({ ev }) {
  return (
    <li className="rounded-lg border border-neutral-200 p-3 dark:border-neutral-800">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <Badge tone="purple">{ev.sub_question_id}</Badge>
        <span className="rounded bg-neutral-900 px-2 py-0.5 font-mono text-white dark:bg-neutral-100 dark:text-neutral-900">
          {ev.doc_id}
        </span>
        <span className="font-mono text-neutral-400">{ev.chunk_id}</span>
        <span className="ml-auto font-mono text-neutral-500">score: {ev.relevance_score}</span>
      </div>
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
        {ev.text}
      </p>
    </li>
  )
}

function ResearcherCard({ update }) {
  return (
    <div className="space-y-3">
      {update.evidence?.length > 0 && (
        <ul className="space-y-2">
          {update.evidence.map((ev) => (
            <EvidenceRow key={ev.chunk_id + ev.sub_question_id} ev={ev} />
          ))}
        </ul>
      )}
      {update.unanswered_sub_questions?.length > 0 && (
        <div className="text-sm text-amber-700 dark:text-amber-400">
          No evidence found for: {update.unanswered_sub_questions.join(', ')}
        </div>
      )}
      {update.refused && (
        <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          {update.final_answer}
        </div>
      )}
    </div>
  )
}

function ClaimRow({ claim }) {
  return (
    <li className="rounded-lg border border-neutral-200 p-3 dark:border-neutral-800">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <Badge tone="purple">{claim.sub_question_id}</Badge>
        <span className="font-mono text-neutral-400">{claim.id}</span>
        <span className="ml-auto font-mono text-neutral-500">confidence: {claim.confidence}</span>
      </div>
      <p className="text-sm text-neutral-800 dark:text-neutral-200">{claim.text}</p>
      <p className="mt-1 text-xs text-neutral-500">citations: {claim.citations.join(', ')}</p>
    </li>
  )
}

function SynthesiserCard({ update }) {
  if (!update.claims?.length) {
    return <p className="text-sm text-neutral-500">No claims produced (no evidence supported any sub-question).</p>
  }
  return (
    <ul className="space-y-2">
      {update.claims.map((claim) => (
        <ClaimRow key={claim.id} claim={claim} />
      ))}
    </ul>
  )
}

const NODE_CARDS = { clarifier: ClarifierCard, planner: PlannerCard, researcher: ResearcherCard, synthesiser: SynthesiserCard }

function StepCard({ step, index }) {
  const Card = NODE_CARDS[step.node]
  return (
    <li className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
      <div className="mb-3 flex items-center gap-2">
        <span className="rounded-full bg-neutral-900 px-2.5 py-0.5 text-xs font-semibold text-white dark:bg-neutral-100 dark:text-neutral-900">
          {index + 1}
        </span>
        <h3 className="font-mono text-sm font-semibold uppercase tracking-wide text-neutral-700 dark:text-neutral-300">
          {step.node}
        </h3>
      </div>
      {Card ? <Card update={step.update} /> : (
        <pre className="overflow-x-auto text-xs text-neutral-600 dark:text-neutral-400">
          {JSON.stringify(step.update, null, 2)}
        </pre>
      )}
    </li>
  )
}

function App() {
  const [baseMermaid, setBaseMermaid] = useState(null)
  const [question, setQuestion] = useState('')
  const [interactive, setInteractive] = useState(true)
  const [threadId, setThreadId] = useState(null)
  const [steps, setSteps] = useState([])
  const [status, setStatus] = useState('idle') // idle | interrupted | done
  const [clarifyingQuestion, setClarifyingQuestion] = useState(null)
  const [reply, setReply] = useState('')
  const [finalState, setFinalState] = useState(null)
  const [showRaw, setShowRaw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/topology')
      .then((res) => res.json())
      .then((body) => setBaseMermaid(body.mermaid))
      .catch(() => setBaseMermaid(null))
  }, [])

  const applyResponse = (body) => {
    setThreadId(body.thread_id)
    setSteps((prev) => [...prev, ...body.steps])
    setStatus(body.status)
    if (body.status === 'interrupted') {
      setClarifyingQuestion(body.clarifying_question)
      setFinalState(null)
    } else {
      setClarifyingQuestion(null)
      setFinalState(body.state)
    }
  }

  const handleRun = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    setSteps([])
    setFinalState(null)
    setClarifyingQuestion(null)
    setStatus('idle')
    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, interactive }),
      })
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`)
      applyResponse(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleResume = async (e) => {
    e.preventDefault()
    if (!reply.trim() || !threadId) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, reply }),
      })
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`)
      setReply('')
      applyResponse(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const doneNodes = new Set(steps.map((s) => s.node))

  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <div className="mx-auto max-w-3xl px-6 py-10">
        <h1 className="text-2xl font-semibold">Research Desk &mdash; Agent Graph</h1>
        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
          Runs Clarifier &rarr; Planner &rarr; Researcher &rarr; Synthesiser one node at a
          time so each node's actual output can be checked, in either interactive (asks a
          follow-up in the browser) or non-interactive (auto-assumes) mode.
        </p>

        <div className="mt-6 rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
          {baseMermaid ? (
            <TopologyDiagram baseMermaid={baseMermaid} doneNodes={doneNodes} />
          ) : (
            <p className="text-sm text-neutral-500">Loading graph topology&hellip;</p>
          )}
        </div>

        <form onSubmit={handleRun} className="mt-6 space-y-3">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. How many days of sick leave am I entitled to?"
            rows={2}
            className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500 dark:border-neutral-700 dark:bg-neutral-900"
          />
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 text-sm text-neutral-600 dark:text-neutral-400">
              <input
                type="checkbox"
                checked={interactive}
                onChange={(e) => setInteractive(e.target.checked)}
              />
              Interactive (pause on a clarifying question)
            </label>
            <button
              type="submit"
              disabled={loading}
              className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 disabled:opacity-50 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-neutral-300"
            >
              {loading ? 'Running…' : 'Run'}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-4 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            {error}
          </div>
        )}

        {steps.length > 0 && (
          <ul className="mt-6 flex flex-col gap-3">
            {steps.map((step, i) => (
              <StepCard key={i} step={step} index={i} />
            ))}
          </ul>
        )}

        {status === 'interrupted' && (
          <form
            onSubmit={handleResume}
            className="mt-4 space-y-2 rounded-lg border border-amber-300 bg-amber-50 p-4 dark:border-amber-900 dark:bg-amber-950"
          >
            <p className="text-sm font-medium text-amber-900 dark:text-amber-200">
              Clarifier is asking: {clarifyingQuestion}
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={reply}
                onChange={(e) => setReply(e.target.value)}
                placeholder="Your answer"
                className="flex-1 rounded-md border border-amber-300 px-3 py-2 text-sm outline-none dark:border-amber-800 dark:bg-neutral-900"
              />
              <button
                type="submit"
                disabled={loading}
                className="rounded-md bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50"
              >
                {loading ? 'Resuming…' : 'Reply'}
              </button>
            </div>
          </form>
        )}

        {status === 'done' && finalState && (
          <div className="mt-6 rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-500">Run finished</h2>
            <dl className="mt-2 grid grid-cols-2 gap-2 text-sm">
              <dt className="text-neutral-500">Retrieval needed</dt>
              <dd>{String(finalState.retrieval_needed)}</dd>
              <dt className="text-neutral-500">Refused</dt>
              <dd>{String(finalState.refused)}</dd>
              {finalState.final_answer && (
                <>
                  <dt className="text-neutral-500">Final answer</dt>
                  <dd>{finalState.final_answer}</dd>
                </>
              )}
            </dl>
            <button
              type="button"
              onClick={() => setShowRaw((v) => !v)}
              className="mt-3 text-xs font-medium text-neutral-500 underline"
            >
              {showRaw ? 'Hide' : 'Show'} raw final state (for debugging)
            </button>
            {showRaw && (
              <pre className="mt-2 max-h-96 overflow-auto rounded bg-neutral-50 p-3 text-xs dark:bg-neutral-900">
                {JSON.stringify(finalState, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
