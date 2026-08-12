import { useEffect, useState } from 'react'

function StatusBar({ status, onReindex, reindexing }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm text-neutral-600 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-400">
      <span>
        {status
          ? `${status.indexed_chunks} chunks indexed • model: ${status.embedding_model}`
          : 'Loading index status…'}
      </span>
      <button
        type="button"
        onClick={onReindex}
        disabled={reindexing}
        className="shrink-0 rounded-md border border-neutral-300 px-3 py-1 text-xs font-medium text-neutral-700 hover:bg-neutral-100 disabled:opacity-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
      >
        {reindexing ? 'Rebuilding…' : 'Rebuild index'}
      </button>
    </div>
  )
}

function ResultCard({ result, rank }) {
  return (
    <li className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded bg-neutral-900 px-2 py-0.5 font-mono text-white dark:bg-neutral-100 dark:text-neutral-900">
          #{rank}
        </span>
        <span className="rounded bg-purple-100 px-2 py-0.5 font-medium text-purple-800 dark:bg-purple-900/40 dark:text-purple-300">
          {result.doc_id}
        </span>
        <span className="font-mono text-neutral-400">{result.chunk_id}</span>
        <span className="ml-auto font-mono text-neutral-500">
          score: {result.relevance_score}
        </span>
      </div>
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
        {result.text}
      </p>
    </li>
  )
}

function App() {
  const [status, setStatus] = useState(null)
  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [reindexing, setReindexing] = useState(false)

  const refreshStatus = async () => {
    try {
      const res = await fetch('/api/status')
      setStatus(await res.json())
    } catch {
      setStatus(null)
    }
  }

  useEffect(() => {
    refreshStatus()
  }, [])

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${res.status})`)
      }
      setResults(await res.json())
    } catch (err) {
      setError(err.message)
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  const handleReindex = async () => {
    setReindexing(true)
    setError(null)
    try {
      const res = await fetch('/api/reindex', { method: 'POST' })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Reindex failed (${res.status})`)
      }
      setStatus(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setReindexing(false)
    }
  }

  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <div className="mx-auto max-w-3xl px-6 py-10">
        <h1 className="text-2xl font-semibold">Research Desk &mdash; Retrieval Test</h1>
        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
          Hybrid search (BGE embeddings + BM25) over the NimbusWorks HR policy docs. No LLM
          involved &mdash; this only exercises retrieve().
        </p>

        <div className="mt-6">
          <StatusBar status={status} onReindex={handleReindex} reindexing={reindexing} />
        </div>

        <form onSubmit={handleSearch} className="mt-6 flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. What is the current sick leave allowance?"
            className="flex-1 rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500 dark:border-neutral-700 dark:bg-neutral-900"
          />
          <input
            type="number"
            min={1}
            max={20}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            title="Number of results (k)"
            className="w-16 rounded-md border border-neutral-300 px-2 py-2 text-sm outline-none focus:border-neutral-500 dark:border-neutral-700 dark:bg-neutral-900"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 disabled:opacity-50 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-neutral-300"
          >
            {loading ? 'Searching…' : 'Search'}
          </button>
        </form>

        {error && (
          <div className="mt-4 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            {error}
          </div>
        )}

        {results && (
          <ul className="mt-6 flex flex-col gap-3">
            {results.length === 0 ? (
              <li className="text-sm text-neutral-500">No results.</li>
            ) : (
              results.map((r, i) => <ResultCard key={r.chunk_id} result={r} rank={i + 1} />)
            )}
          </ul>
        )}
      </div>
    </div>
  )
}

export default App
