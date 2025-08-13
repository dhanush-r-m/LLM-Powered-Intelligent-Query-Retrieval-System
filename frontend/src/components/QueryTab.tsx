import { useState } from 'react'
import toast from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

type SourceRef = {
  content: string
  score: number
  metadata?: { filename?: string, page_number?: number, line_range?: [number, number], clause_id?: string }
}

type QueryResult = {
  question: string
  response: string
  sources: SourceRef[]
  timestamp: string
  confidence?: number
}

export default function QueryTab({ apiBase }: { apiBase: string }) {
  const [question, setQuestion] = useState('')
  const [nResults, setNResults] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResult | null>(null)

  async function submit() {
    if (!question.trim()) return toast.error('Please enter a question')
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch(`${apiBase}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, n_results: nResults })
      })
      if (res.status === 400) {
        const err = await res.json()
        toast.error(err.detail || 'Please upload documents first.')
        return
      }
      const data: QueryResult = await res.json()
      setResult(data)
    } catch (e: any) {
      toast.error(e.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  function copyAnswer() {
    if (!result) return
    navigator.clipboard.writeText(result.response)
    toast.success('Answer copied to clipboard')
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Ask a question</h2>
      <div className="flex flex-col gap-3">
        <input
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') submit() }}
          placeholder="Ask a question about the uploaded documents..."
          className="w-full rounded-xl border border-slate-300 dark:border-slate-600 bg-white/60 dark:bg-white/5 backdrop-blur px-4 py-3 text-base"
        />
        <div className="flex items-center gap-4">
          <label className="text-sm text-slate-600 dark:text-slate-300">n_results: {nResults}</label>
          <input type="range" min={1} max={10} value={nResults} onChange={e => setNResults(parseInt(e.target.value))} />
          <button onClick={submit} className="ml-auto px-4 py-2 rounded-lg bg-primary-600 text-white">Search</button>
        </div>
      </div>

      <div className="mt-6">
        <AnimatePresence>
          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="p-6 rounded-xl bg-white/60 dark:bg-white/5 backdrop-blur">
              <div className="animate-pulse space-y-4">
                <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-3/4" />
                <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-5/6" />
                <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-2/3" />
              </div>
              <div className="mt-4 text-slate-500">Thinking<span className="dots" /></div>
            </motion.div>
          )}
        </AnimatePresence>

        {result && (
          <motion.div initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="rounded-xl p-6 bg-white/60 dark:bg-white/5 backdrop-blur">
            <div className="flex items-start gap-3">
              <div className="text-2xl">🤖</div>
              <div className="flex-1">
                <div className="flex items-center gap-3">
                  <h3 className="font-semibold">Answer</h3>
                  {typeof result.confidence === 'number' && (
                    <span className="text-xs text-slate-600 dark:text-slate-400">Confidence: {(result.confidence * 100).toFixed(0)}%</span>
                  )}
                </div>
                <p className="mt-2 whitespace-pre-wrap">{result.response}</p>
                <div className="mt-3">
                  <button onClick={copyAnswer} className="text-sm px-3 py-1.5 rounded bg-slate-800 text-white dark:bg-slate-200 dark:text-slate-900">Copy</button>
                </div>
              </div>
            </div>

            <div className="mt-5">
              <h4 className="font-medium mb-2">References</h4>
              <div className="space-y-2">
                {result.sources.length === 0 && (
                  <div className="text-sm text-slate-500">No references returned.</div>
                )}
                {result.sources.map((s, i) => (
                  <details key={i} className="group bg-white/70 dark:bg-white/5 backdrop-blur rounded-lg p-3">
                    <summary className="cursor-pointer list-none flex items-center justify-between">
                      <div className="font-medium truncate">{s.metadata?.clause_id || s.metadata?.filename || 'Clause'}</div>
                      <div className="text-xs text-slate-500">Score: {(s.score * 100).toFixed(1)}%</div>
                    </summary>
                    <div className="mt-2 text-sm text-slate-700 dark:text-slate-200 whitespace-pre-wrap">{s.content}</div>
                    <div className="mt-1 text-xs text-slate-500">Source: {s.metadata?.filename || 'Unknown'} {s.metadata?.page_number ? `(p.${s.metadata.page_number})` : ''}</div>
                  </details>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

