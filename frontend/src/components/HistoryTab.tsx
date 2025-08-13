import { useEffect, useState } from 'react'
import toast from 'react-hot-toast'

type HistoryItem = {
  timestamp: string
  question: string
  answer: string
}

export default function HistoryTab({ apiBase }: { apiBase: string }) {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [confirmOpen, setConfirmOpen] = useState(false)

  async function load() {
    try {
      const res = await fetch(`${apiBase}/history`)
      const data: any[] = await res.json()
      setItems(data.map(d => ({ timestamp: d.timestamp, question: d.question, answer: d.answer })))
    } catch (e: any) {
      toast.error(e.message || 'Failed to load history')
    }
  }

  async function clear() {
    try {
      await fetch(`${apiBase}/clear-history`, { method: 'POST' })
      setItems([])
      toast.success('History cleared')
    } catch (e: any) {
      toast.error(e.message || 'Failed to clear history')
    } finally {
      setConfirmOpen(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Query History</h2>
        <div className="flex gap-2">
          <button onClick={load} className="px-3 py-2 rounded-lg bg-slate-200 dark:bg-white/10">Refresh</button>
          <button onClick={() => setConfirmOpen(true)} className="px-3 py-2 rounded-lg bg-red-600 text-white">Clear History</button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="text-slate-500">No history yet.</div>
      ) : (
        <ul className="space-y-3">
          {items.map((it, i) => (
            <li key={i} className="bg-white/60 dark:bg-white/5 backdrop-blur rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="font-medium">{it.question}</div>
                <div className="text-xs text-slate-500">{new Date(it.timestamp).toLocaleString()}</div>
              </div>
              <div className="mt-1 text-sm text-slate-600 dark:text-slate-300 line-clamp-2">{it.answer}</div>
            </li>
          ))}
        </ul>
      )}

      {confirmOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-900 rounded-xl p-6 max-w-sm w-full">
            <h3 className="font-semibold text-lg">Clear history?</h3>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">This action cannot be undone.</p>
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => setConfirmOpen(false)} className="px-3 py-2 rounded-lg bg-slate-200 dark:bg-white/10">Cancel</button>
              <button onClick={clear} className="px-3 py-2 rounded-lg bg-red-600 text-white">Clear</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

