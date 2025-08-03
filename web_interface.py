from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from pathlib import Path



app = FastAPI(title="Intelligent Query-Retrieval System", version="1.0.0")



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
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .query-section {
                margin-bottom: 30px;
            }
            .query-input {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                margin-bottom: 15px;
            }
            .btn {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
            }
            .btn:hover {
                background-color: #0056b3;
            }
            .btn-secondary {
                background-color: #6c757d;
            }
            .btn-secondary:hover {
                background-color: #545b62;
            }
            .response-section {
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                display: none;
            }
            .source {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #007bff;
                border-radius: 4px;
            }
            .source-score {
                color: #666;
                font-size: 12px;
                float: right;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .document-upload {
                border: 2px dashed #ddd;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
            }
            .tab.active {
                border-bottom-color: #007bff;
                color: #007bff;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Intelligent Query-Retrieval System</h1>
            
            <div class="tabs">
                <div class="tab active" onclick="showTab('query')">Query</div>
                <div class="tab" onclick="showTab('documents')">Add Documents</div>
                <div class="tab" onclick="showTab('history')">History</div>
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
                    <h3>Add Documents to Knowledge Base</h3>
                    <textarea id="documentText" 
                              placeholder="Paste your documents here..." 
                              style="width: 100%; height: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 4px;"></textarea>
                    <br><br>
                    <button class="btn" onclick="addDocument()">Add Document</button>
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
            function showTab(tabName) {
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
                event.target.classList.add('active');
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
                            <div class="source-score">Score: ${(source.score * 100).toFixed(1)}%</div>
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
            
            async function addDocument() {
                const text = document.getElementById('documentText').value.trim();
                if (!text) {
                    alert('Please enter some text');
                    return;
                }
                
                try {
                    const response = await fetch('/add-documents', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            texts: [text],
                            metadatas: [{"source": "user_input", "added_via": "web_interface"}]
                        })
                    });
                    
                    const data = await response.json();
                    document.getElementById('documentStatus').innerHTML = 
                        '<div style="color: green; padding: 10px;">✅ Document added successfully!</div>';
                    document.getElementById('documentText').value = '';
                    
                    setTimeout(() => {
                        document.getElementById('documentStatus').innerHTML = '';
                    }, 3000);
                    
                } catch (error) {
                    document.getElementById('documentStatus').innerHTML = 
                        '<div style="color: red; padding: 10px;">❌ Error adding document: ' + error.message + '</div>';
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
                            <p><strong>A:</strong> ${item.response}</p>
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
        </script>
    </body>
    </html>
    """

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the retrieval system"""
    try:
        return {
            "question": request.question,
            "response": f"This is a mock response for: {request.question}. Please integrate with the actual IntelligentRetrieval class.",
            "sources": [
                {
                    "content": f"Sample relevant content for {request.question}...",
                    "score": 0.85,
                    "metadata": {"filename": "sample.txt"}
                }
            ],
            "timestamp": "2024-01-01T12:00:00"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-documents")
async def add_documents(request: DocumentRequest):
    """Add documents to the knowledge base"""
    try:
        return {"message": f"Added {len(request.texts)} documents successfully (mock)"}
        
           
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    """Get query history"""
    try:
        return []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history")
async def clear_history():
    """Clear query history"""
    try:
        return {"message": "History cleared (mock)"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
