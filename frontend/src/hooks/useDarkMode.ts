import { useEffect, useState } from 'react'

export function useDarkMode() {
  const [isDark, setIsDark] = useState<boolean>(() => {
    const stored = localStorage.getItem('dark-mode')
    if (stored !== null) return stored === '1'
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    const root = document.documentElement
    if (isDark) root.classList.add('dark')
    else root.classList.remove('dark')
    localStorage.setItem('dark-mode', isDark ? '1' : '0')
  }, [isDark])

  return { isDark, toggle: () => setIsDark(v => !v) }
}

