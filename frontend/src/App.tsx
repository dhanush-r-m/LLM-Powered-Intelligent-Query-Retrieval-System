import { useMemo, useState } from 'react'
import Navbar from './components/Navbar'
import DragDropUpload from './components/DragDropUpload'
import QueryTab from './components/QueryTab'
import HistoryTab from './components/HistoryTab'

type TabKey = 'documents' | 'query' | 'history'

function getApiBase() {
  return (import.meta.env.VITE_API_BASE as string) || ''
}

export default function App() {
  const [active, setActive] = useState<TabKey>('query')
  const apiBase = useMemo(getApiBase, [])

  return (
    <div className="min-h-screen text-slate-800 dark:text-slate-100">
      <div className="fixed inset-0 -z-10 bg-gradient-to-br from-blue-100 via-white to-purple-100 dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950" />
      <div className="max-w-6xl mx-auto px-4 py-6">
        <Navbar active={active} onChange={setActive} />

        <div className="mt-6">
          {active === 'documents' && (
            <div className="glass-card">
              <DragDropUpload apiBase={apiBase} />
            </div>
          )}
          {active === 'query' && (
            <div className="glass-card">
              <QueryTab apiBase={apiBase} />
            </div>
          )}
          {active === 'history' && (
            <div className="glass-card">
              <HistoryTab apiBase={apiBase} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

