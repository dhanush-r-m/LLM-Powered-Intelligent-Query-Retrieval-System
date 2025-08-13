import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'
import { Toaster } from 'react-hot-toast'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
    <Toaster position="top-right" toastOptions={{
      style: { fontFamily: 'Inter, system-ui, Avenir, Helvetica, Arial, sans-serif' }
    }} />
  </React.StrictMode>
)

