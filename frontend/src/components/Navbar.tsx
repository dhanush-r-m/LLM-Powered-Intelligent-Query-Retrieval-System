import { motion } from 'framer-motion'
import { useDarkMode } from '../hooks/useDarkMode'

type TabKey = 'documents' | 'query' | 'history'

export default function Navbar({ active, onChange }: {
  active: TabKey,
  onChange: (t: TabKey) => void
}) {
  const tabs: { key: TabKey, label: string }[] = [
    { key: 'query', label: 'Query' },
    { key: 'documents', label: 'Documents' },
    { key: 'history', label: 'History' },
  ]
  const { isDark, toggle } = useDarkMode()

  return (
    <div className="flex items-center justify-between">
      <div className="relative flex bg-white/60 dark:bg-white/5 rounded-xl p-1 shadow-glass backdrop-blur">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={`relative z-10 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${active === t.key ? 'text-slate-900 dark:text-white' : 'text-slate-600 dark:text-slate-300'}`}
          >
            {t.label}
            {active === t.key && (
              <motion.span layoutId="tab-indicator" className="absolute inset-0 -z-10 rounded-lg bg-gradient-to-r from-blue-500/20 to-purple-500/20 dark:from-indigo-400/20 dark:to-fuchsia-400/20" />
            )}
          </button>
        ))}
      </div>

      <button onClick={toggle} className="px-3 py-2 rounded-lg bg-white/60 dark:bg-white/5 shadow-glass backdrop-blur text-sm">
        {isDark ? '🌙 Dark' : '☀️ Light'}
      </button>
    </div>
  )
}

