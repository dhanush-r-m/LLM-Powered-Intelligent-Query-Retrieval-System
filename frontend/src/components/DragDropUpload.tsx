import { useRef, useState } from 'react'
import toast from 'react-hot-toast'

export default function DragDropUpload({ apiBase }: { apiBase: string }) {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function onDrop(ev: React.DragEvent) {
    ev.preventDefault()
    const list = Array.from(ev.dataTransfer.files).filter(f => f.type === 'application/pdf')
    if (list.length === 0) return toast.error('Only PDF files are supported')
    setFiles(prev => [...prev, ...list])
  }
  function onDragOver(ev: React.DragEvent) {
    ev.preventDefault()
  }

  function onPick(ev: React.ChangeEvent<HTMLInputElement>) {
    const list = Array.from(ev.target.files || []).filter(f => f.type === 'application/pdf')
    if (list.length === 0) return toast.error('Only PDF files are supported')
    setFiles(prev => [...prev, ...list])
  }

  function removeAt(index: number) {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  async function upload() {
    if (files.length === 0) return toast.error('Please select at least one PDF')
    setUploading(true)
    try {
      const fd = new FormData()
      files.forEach(f => fd.append('files', f))
      const res = await fetch(`${apiBase}/upload-pdf`, { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      toast.success(data.message || 'FAISS index rebuilt successfully')
      setFiles([])
    } catch (e: any) {
      toast.error(e.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Add PDFs to Knowledge Base</h2>

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-8 text-center bg-white/60 dark:bg-white/5 backdrop-blur hover:border-primary-600 transition-colors"
      >
        <div className="text-4xl">📄</div>
        <p className="mt-2">Drag and drop PDF files here, or</p>
        <button
          className="mt-3 px-4 py-2 rounded-lg bg-primary-600 text-white shadow-glass"
          onClick={() => inputRef.current?.click()}
        >Choose Files</button>
        <input ref={inputRef} type="file" multiple accept="application/pdf" className="hidden" onChange={onPick} />
      </div>

      {files.length > 0 && (
        <div className="mt-6">
          <h3 className="font-medium mb-2">Selected files</h3>
          <ul className="space-y-2">
            {files.map((f, i) => (
              <li key={i} className="flex items-center justify-between bg-white/70 dark:bg-white/5 backdrop-blur rounded-lg px-4 py-2">
                <div className="truncate">
                  <span className="font-medium">{f.name}</span>
                  <span className="text-slate-500 dark:text-slate-400 ml-2 text-sm">({(f.size/1024/1024).toFixed(2)} MB)</span>
                </div>
                <button className="text-red-600 hover:underline" onClick={() => removeAt(i)}>Remove</button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-6 flex items-center gap-3">
        <button onClick={upload} disabled={uploading || files.length === 0} className="px-4 py-2 rounded-lg bg-primary-600 text-white disabled:opacity-50">
          {uploading ? 'Uploading…' : 'Upload & Build Index'}
        </button>
        {uploading && (
          <div className="flex-1 h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 animate-progress" />
          </div>
        )}
      </div>
    </div>
  )
}

