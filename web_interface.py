from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from pathlib import Path
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import aiofiles

from intelligent_query_system import IntelligentQuerySystem



load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not found in environment variables. Please set it in .env")

app = FastAPI(title="Intelligent Query-Retrieval System", version="1.0.0")

# Enable CORS for local dev of SPA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
DOCUMENTS_DIR = Path("Documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("query_system_cache")
CACHE_DIR.mkdir(exist_ok=True)
VECTOR_INDEX_DIR = CACHE_DIR / "vector_index"

# Core engine
query_system = IntelligentQuerySystem(groq_api_key=groq_api_key, cache_dir=str(CACHE_DIR))

# Simple in-process flag to prevent concurrent index builds
index_build_lock = asyncio.Lock()

def index_exists() -> bool:
    index_path = VECTOR_INDEX_DIR / "index.faiss"
    chunks_path = VECTOR_INDEX_DIR / "chunks.pkl"
    embeddings_path = VECTOR_INDEX_DIR / "embeddings.pkl"
    return index_path.exists() and chunks_path.exists() and embeddings_path.exists()

async def ensure_index_loaded():
    # If index already present in memory, nothing to do
    if getattr(query_system.vector_db, "index", None) is not None:
        return
    # Otherwise, try loading from disk
    if index_exists():
        query_system.load_knowledge_base(str(VECTOR_INDEX_DIR))
    else:
        raise HTTPException(status_code=400, detail="No knowledge base found. Upload PDFs to build the index.")



class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 5

class DocumentRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Intelligent Query-Retrieval System</title>
        <style>
            :root { --bg: #f5f7ff; --card: rgba(255,255,255,0.85); --text: #111827; --muted:#6b7280; --primary:#6366f1; }
            body.dark { --bg: #0b1020; --card: rgba(17,24,39,0.7); --text: #e5e7eb; --muted:#9ca3af; --primary:#818cf8; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: radial-gradient(1000px 500px at 20% 0%, #dbeafe, transparent),
                            radial-gradient(800px 400px at 80% 0%, #f5d0fe, transparent), var(--bg);
                color: var(--text);
                transition: background 0.3s ease, color 0.3s ease;
            }
            .container {
                background: var(--card);
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(2,6,23,0.15);
                margin-bottom: 20px;
                backdrop-filter: blur(8px);
            }
            h1 {
                text-align: center;
                margin-bottom: 20px;
                font-weight: 800;
                background: linear-gradient(90deg, #6366f1, #22d3ee, #a78bfa);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }
            .brand { display:flex; align-items:center; justify-content:center; gap:14px; }
            .brand h1 { margin:0; line-height:1.2; padding-bottom:4px; display:inline-block; }
            .logo { width:48px; height:48px; filter: drop-shadow(0 4px 10px rgba(2,6,23,0.35)); border-radius:12px; }
            .query-section {
                margin-bottom: 30px;
            }
            .query-input {
                width: 100%;
                padding: 15px;
                border: 2px solid #d1d5db;
                border-radius: 8px;
                font-size: 16px;
                margin-bottom: 15px;
                background: transparent;
                color: var(--text);
            }
            .btn {
                background: linear-gradient(135deg, var(--primary), #22d3ee);
                color: #fff;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
                box-shadow: 0 6px 14px rgba(99,102,241,0.35);
            }
            .btn:hover {
                filter: brightness(1.05);
            }
            .btn-secondary {
                background: #374151;
            }
            .btn-secondary:hover {
                background-color: #1f2937;
            }
            .response-section {
                margin-top: 20px;
                padding: 20px;
                background-color: rgba(203,213,225,0.15);
                border-radius: 8px;
                display: none;
            }
            .source {
                background: var(--card);
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid var(--primary);
                border-radius: 4px;
            }
            .source-score {
                color: var(--muted);
                font-size: 12px;
                float: right;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .document-upload {
                border: 2px dashed #94a3b8;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 1px solid #cbd5e1;
                gap: 8px;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                border-radius: 8px 8px 0 0;
                color: var(--muted);
            }
            .tab.active {
                border-bottom-color: var(--primary);
                color: var(--text);
                font-weight: 600;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="brand">
                <svg class="logo" viewBox="0 0 96 96" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <defs>
                        <linearGradient id="lg1" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#6366f1"/>
                            <stop offset="50%" stop-color="#06b6d4"/>
                            <stop offset="100%" stop-color="#a78bfa"/>
                        </linearGradient>
                        <linearGradient id="lg2" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="#ffffff" stop-opacity="0.9"/>
                            <stop offset="100%" stop-color="#ffffff" stop-opacity="0.7"/>
                        </linearGradient>
                    </defs>
                    <rect x="8" y="14" width="80" height="60" rx="18" fill="url(#lg1)"/>
                    <rect x="8" y="14" width="80" height="60" rx="18" fill="#000" opacity="0.08"/>
                    <circle cx="34" cy="42" r="8" fill="url(#lg2)"/>
                    <circle cx="62" cy="42" r="8" fill="url(#lg2)"/>
                    <rect x="32" y="56" width="32" height="6" rx="3" fill="#ffffffcc"/>
                    <rect x="46" y="6" width="4" height="10" rx="2" fill="url(#lg1)"/>
                    <circle cx="24" cy="18" r="4" fill="url(#lg1)"/>
                    <circle cx="72" cy="18" r="4" fill="url(#lg1)"/>
                </svg>
                <h1>Intelligent Query-Retrieval System</h1>
            </div>
            <div class="topbar">
                <div></div>
                <button id="themeToggle" class="btn btn-secondary" onclick="toggleTheme()">Toggle Theme</button>
            </div>
            <div class="tabs">
                <div class="tab active" onclick="showTab('query', this)">Query</div>
                <div class="tab" onclick="showTab('documents', this)">Add Documents</div>
                <div class="tab" onclick="showTab('history', this)">History</div>
            </div>
            
            <!-- Query Tab -->
            <div id="query-tab" class="tab-content active">
                <div class="query-section">
                    <input type="text" 
                           id="queryInput" 
                           class="query-input" 
                           placeholder="Ask a question..."
                           onkeypress="handleKeyPress(event)">
                    <button class="btn" onclick="submitQuery()">Search</button>
                    <button class="btn btn-secondary" onclick="clearResults()">Clear</button>
                </div>
                
                <div class="loading" id="loading">
                    <p>🔍 Searching for relevant information...</p>
                </div>
                
                <div class="response-section" id="responseSection">
                    <h3>Response</h3>
                    <div id="response"></div>
                    <h4>Sources</h4>
                    <div id="sources"></div>
                </div>
            </div>
            
            <!-- Documents Tab -->
            <div id="documents-tab" class="tab-content">
                <div class="document-upload">
                    <h3>Add PDFs to Knowledge Base</h3>
                    <input type="file" id="pdfFiles" accept="application/pdf" multiple />
                    <p style="color:#666; font-size: 13px;">Only PDF files are supported.</p>
                    <button class="btn" onclick="uploadPDFs()">Upload & Build Index</button>
                </div>

                <div id="documentStatus"></div>
            </div>
            
            <!-- History Tab -->
            <div id="history-tab" class="tab-content">
                <h3>Query History</h3>
                <button class="btn btn-secondary" onclick="loadHistory()">Refresh History</button>
                <button class="btn btn-secondary" onclick="clearHistory()">Clear History</button>
                <div id="historyContent"></div>
            </div>
        </div>

        <script>
            function showTab(tabName, el) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabName + '-tab').classList.add('active');
                
                // Add active class to clicked tab
                if (el) { el.classList.add('active'); }

                if (tabName === 'history') { loadHistory(); }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    submitQuery();
                }
            }
            
            async function submitQuery() {
                const query = document.getElementById('queryInput').value.trim();
                if (!query) {
                    alert('Please enter a question');
                    return;
                }
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('responseSection').style.display = 'none';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: query,
                            n_results: 5
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    
                    // Show results
                    displayResults(data);
                    
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    alert('Error: ' + error.message);
                }
            }
            
            function displayResults(data) {
                const responseDiv = document.getElementById('response');
                const sourcesDiv = document.getElementById('sources');
                
                responseDiv.innerHTML = `<p><strong>Answer:</strong> ${data.response}</p>`;
                
                sourcesDiv.innerHTML = '';
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach((source, index) => {
                        const sourceDiv = document.createElement('div');
                        sourceDiv.className = 'source';
                        sourceDiv.innerHTML = `
                            <p>${source.content}</p>
                            <small>Source: ${source.metadata.filename || 'Unknown'}</small>
                        `;
                        sourcesDiv.appendChild(sourceDiv);
                    });
                } else {
                    sourcesDiv.innerHTML = '<p>No sources found.</p>';
                }
                
                document.getElementById('responseSection').style.display = 'block';
            }
            
            function clearResults() {
                document.getElementById('queryInput').value = '';
                document.getElementById('responseSection').style.display = 'none';
                document.getElementById('loading').style.display = 'none';
            }
            
            async function uploadPDFs() {
                const input = document.getElementById('pdfFiles');
                if (!input.files || input.files.length === 0) {
                    alert('Please select one or more PDF files');
                    return;
                }

                const formData = new FormData();
                for (let i = 0; i < input.files.length; i++) {
                    formData.append('files', input.files[i]);
                }

                document.getElementById('documentStatus').innerHTML = '<div style="padding: 10px;">⏳ Uploading and building index...</div>';

                try {
                    const response = await fetch('/upload-pdf', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Upload failed');
                    }

                    document.getElementById('documentStatus').innerHTML = 
                        '<div style="color: green; padding: 10px;">✅ ' + (data.message || 'Index built successfully!') + '</div>';
                    input.value = '';
                } catch (error) {
                    document.getElementById('documentStatus').innerHTML = 
                        '<div style="color: red; padding: 10px;">❌ Error: ' + error.message + '</div>';
                }
            }
            
            async function loadHistory() {
                try {
                    const response = await fetch('/history');
                    const data = await response.json();
                    
                    const historyContent = document.getElementById('historyContent');
                    historyContent.innerHTML = '';
                    
                    if (data.length === 0) {
                        historyContent.innerHTML = '<p>No queries in history.</p>';
                        return;
                    }
                    
                    data.forEach((item, index) => {
                        const historyItem = document.createElement('div');
                        historyItem.className = 'source';
                        historyItem.innerHTML = `
                            <p><strong>Q${index + 1}:</strong> ${item.question}</p>
                            <p><strong>A:</strong> ${item.answer || item.response || ''}</p>
                            <small>Time: ${new Date(item.timestamp).toLocaleString()}</small>
                        `;
                        historyContent.appendChild(historyItem);
                    });
                    
                } catch (error) {
                    alert('Error loading history: ' + error.message);
                }
            }
            
            async function clearHistory() {
                try {
                    const response = await fetch('/clear-history', {method: 'POST'});
                    document.getElementById('historyContent').innerHTML = '<p>History cleared.</p>';
                } catch (error) {
                    alert('Error clearing history: ' + error.message);
                }
            }
            // Theme handling
            (function initTheme(){
                try {
                    const saved = localStorage.getItem('theme');
                    if (saved === 'dark') document.body.classList.add('dark');
                    else if (!saved) {
                        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                            document.body.classList.add('dark');
                        }
                    }
                } catch(e) {}
            })();
            function toggleTheme(){
                document.body.classList.toggle('dark');
                try { localStorage.setItem('theme', document.body.classList.contains('dark') ? 'dark' : 'light'); } catch(e) {}
            }
        </script>
    </body>
    </html>
    """

@app.post("/query")
async def query_documents(request: QueryRequest, raw_request: Request):
    """Query the retrieval system using the built knowledge base"""
    try:
        await ensure_index_loaded()

        # Execute query
        response = query_system.query(request.question, k=int(request.n_results or 5))

        # Transform references for UI
        sources = []
        for ref in response.answer.get("references", []):
            sources.append({
                "content": ref.get("clause", ""),
                "score": ref.get("relevance_score", 0.0),
                "metadata": {
                    "filename": Path(ref.get("source", "")).name or "Unknown",
                    "page_number": ref.get("page_number"),
                    "line_range": ref.get("line_range"),
                    "clause_id": ref.get("clause_id")
                }
            })

        result = {
            "question": response.query,
            "response": response.answer.get("response", ""),
            "sources": sources,
            "timestamp": response.timestamp,
            "confidence": response.answer.get("confidence_score", 0.0),
        }

        # Append to query log
        log_entry = {
            "timestamp": response.timestamp,
            "client_ip": raw_request.client.host if raw_request and raw_request.client else "unknown",
            "question": response.query,
            "answer": response.answer.get("response", ""),
            "confidence": response.answer.get("confidence_score", 0.0),
            "processing_time_ms": response.processing_time_ms,
        }
        try:
            log_path = Path("query_log.json")
            existing: List[dict] = []
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(log_entry)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception:
            # Non-fatal logging failure
            pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve built SPA if present (frontend/dist)
FRONTEND_DIST = Path(__file__).parent / 'frontend' / 'dist'
if FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="app")

@app.post("/upload-pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    """Upload one or more PDFs and rebuild the knowledge base index"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved_paths: List[str] = []
    try:
        # Save uploaded PDFs
        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Unsupported file type for {f.filename}. Only PDFs are allowed.")
            destination = DOCUMENTS_DIR / f.filename
            async with aiofiles.open(destination, "wb") as out_file:
                content = await f.read()
                await out_file.write(content)
            saved_paths.append(str(destination))

        # Build index with ALL PDFs under Documents (guard against concurrent builds)
        async with index_build_lock:
            all_pdf_paths = [str(p) for p in DOCUMENTS_DIR.rglob("*.pdf")]
            if not all_pdf_paths:
                raise HTTPException(status_code=400, detail="No PDFs found after upload")
            await query_system.build_knowledge_base(all_pdf_paths, save_index=True)

        return {"message": f"Uploaded {len(saved_paths)} PDF(s) and built index from {len(all_pdf_paths)} document(s) successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    """Get query history"""
    try:
        log_path = Path("query_log.json")
        if not log_path.exists():
            return []
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # return the last 50 entries
        return data[-50:]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history")
async def clear_history():
    """Clear query history"""
    try:
        log_path = Path("query_log.json")
        if log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        return {"message": "History cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
