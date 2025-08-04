import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from pathlib import Path
import PyPDF2
import docx
import email
from email import policy
from email.parser import BytesParser
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from groq import Groq
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

load_dotenv()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    source_file: str
    chunk_id: str
    line_number: Optional[int] = None  
    clause_id: Optional[str] = None    
    document_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  

@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with relevance score."""
    chunk: DocumentChunk
    relevance_score: float
    weight: float

@dataclass
class QueryResponse:
    """Structured JSON response for query results."""
    query: str
    answer: Dict[str, Any]
    confidence_score: float
    decision_rationale: str
    timestamp: str
    processing_time_ms: int

class DocumentProcessor:
    """Handles processing of legal documents (PDF, DOCX, EML, TXT)."""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.eml', '.txt'}
    
    def is_clause_heading(self, text: str) -> bool:
        """Check if the text is a clause heading using a regex pattern."""
        pattern = r"^(Section|Clause|Article|Paragraph)\s+[\d\w\.\(\)]+"
        return bool(re.match(pattern, text.strip(), re.IGNORECASE))
    
    def extract_clause_id(self, text: str) -> Optional[str]:
        """Extract the clause identifier from the text."""
        pattern = r"^(Section|Clause|Article|Paragraph)\s+[\d\w\.\(\)]+"
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        return match.group(0) if match else None
    
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from PDF files, grouping into clauses with identifiers."""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        current_clause_lines = []
                        current_clause_id = None
                        start_line = 1
                        for line_num, line in enumerate(lines, 1):
                            if self.is_clause_heading(line):
                                # Save previous clause
                                if current_clause_lines:
                                    clause_text = '\n'.join(current_clause_lines)
                                    clause_id = current_clause_id or f"{Path(file_path).stem}_page_{page_num}_lines_{start_line}-{line_num-1}"
                                    chunks.append(DocumentChunk(
                                        content=clause_text,
                                        source_file=file_path,
                                        chunk_id=clause_id,
                                        line_number=start_line,
                                        clause_id=current_clause_id,
                                        document_type="pdf",
                                        metadata={"page_number": page_num + 1, "line_range": (start_line, line_num - 1)}
                                    ))
                         
                                current_clause_id = self.extract_clause_id(line)
                                current_clause_lines = [line.strip()]
                                start_line = line_num
                            else:
                                current_clause_lines.append(line.strip())
                      
                        if current_clause_lines:
                            clause_text = '\n'.join(current_clause_lines)
                            clause_id = current_clause_id or f"{Path(file_path).stem}_page_{page_num}_lines_{start_line}-{len(lines)}"
                            chunks.append(DocumentChunk(
                                content=clause_text,
                                source_file=file_path,
                                chunk_id=clause_id,
                                line_number=start_line,
                                clause_id=current_clause_id,
                                document_type="pdf",
                                metadata={"page_number": page_num + 1, "line_range": (start_line, len(lines))}
                            ))
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
        return chunks
    
    def process_docx(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from DOCX files, grouping into clauses with identifiers."""
        chunks = []
        try:
            doc = docx.Document(file_path)
            current_clause_paras = []
            current_clause_id = None
            para_num = 0
            for para in doc.paragraphs:
                para_num += 1
                if para.text.strip():
                    if self.is_clause_heading(para.text):
                        if current_clause_paras:
                            clause_text = '\n'.join(current_clause_paras)
                            clause_id = current_clause_id or f"{Path(file_path).stem}_para_{para_num}"
                            chunks.append(DocumentChunk(
                                content=clause_text,
                                source_file=file_path,
                                chunk_id=clause_id,
                                line_number=para_num,
                                clause_id=current_clause_id,
                                document_type="docx",
                                metadata={"paragraph_number": para_num}
                            ))
                        current_clause_id = self.extract_clause_id(para.text)
                        current_clause_paras = [para.text.strip()]
                    else:
                        current_clause_paras.append(para.text.strip())
            # Add the last clause
            if current_clause_paras:
                clause_text = '\n'.join(current_clause_paras)
                clause_id = current_clause_id or f"{Path(file_path).stem}_para_{para_num}"
                chunks.append(DocumentChunk(
                    content=clause_text,
                    source_file=file_path,
                    chunk_id=clause_id,
                    line_number=para_num,
                    clause_id=current_clause_id,
                    document_type="docx",
                    metadata={"paragraph_number": para_num}
                ))
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
        return chunks
    
    def process_email(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from email files with line numbers."""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)
                subject = msg.get('Subject', 'No Subject')
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body += part.get_content() or ""
                else:
                    body = msg.get_content() or ""
                
                lines = body.split('\n')
                for line_num, line in enumerate(lines, 1):
                    if line.strip():
                        clause_id = f"{Path(file_path).stem}_email_line_{line_num}"
                        chunks.append(DocumentChunk(
                            content=line.strip(),
                            source_file=file_path,
                            chunk_id=clause_id,
                            line_number=line_num,
                            clause_id=clause_id,
                            document_type="email",
                            metadata={"subject": subject}
                        ))
        except Exception as e:
            logger.error(f"Error processing email {file_path}: {e}")
        return chunks
    
    def process_text(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from TXT files with line numbers."""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    if line.strip():
                        clause_id = f"{Path(file_path).stem}_line_{line_num}"
                        chunks.append(DocumentChunk(
                            content=line.strip(),
                            source_file=file_path,
                            chunk_id=clause_id,
                            line_number=line_num,
                            clause_id=clause_id,
                            document_type="text",
                            metadata={}
                        ))
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}: {e}")
        return chunks
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document based on its file extension."""
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_ext}")
            return []
        
        if file_ext == '.pdf':
            return self.process_pdf(file_path)
        elif file_ext == '.docx':
            return self.process_docx(file_path)
        elif file_ext == '.eml':
            return self.process_email(file_path)
        elif file_ext == '.txt':
            return self.process_text(file_path)
        return []

class VectorDatabase:
    """Manages vector embeddings and semantic search using FAISS."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.legal_keywords = [
            "clause", "section", "article", "provision", "term", "condition", 
            "disclaimer", "disclosure", "policy", "contract", "agreement", 
            "liability", "obligation", "warranty", "indemnity"
        ]
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for document chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def compute_dynamic_weights(self, chunks: List[DocumentChunk]) -> List[float]:
        """Dynamically assign weights based on legal content relevance."""
        weights = []
        for chunk in chunks:
            content_lower = chunk.content.lower()
            keyword_score = sum(1 for keyword in self.legal_keywords if keyword in content_lower)
            length_score = min(len(chunk.content.split()) / 100, 1.0)
            weight = 0.7 * keyword_score / len(self.legal_keywords) + 0.3 * length_score
            weights.append(max(0.1, min(weight, 1.0)))
        return weights
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index with dynamic weights."""
        logger.info(f"Building vector index for {len(chunks)} chunks...")
        self.chunks = chunks
        self.embeddings = self.create_embeddings(chunks)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info("Vector index built successfully")
    
    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search for relevant chunks using semantic similarity with dynamic weights."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        weights = self.compute_dynamic_weights(self.chunks)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                weighted_score = score * weights[idx]
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    relevance_score=float(weighted_score),
                    weight=weights[idx]
                ))
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:k]
    
    def save_index(self, path: str):
        """Save the FAISS index and metadata."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(path, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load the FAISS index and metadata."""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(path, "embeddings.pkl"), "rb") as f:
            self.embeddings = pickle.load(f)
        logger.info(f"Index loaded from {path}")

class LLMProcessor:
    """Handles LLM interactions for clause extraction and response generation."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = 8192
    
    def generate_response(self, query: str, retrieved_chunks: List[RetrievalResult]) -> QueryResponse:
        """Generate structured JSON response with detailed references and specific details."""
        start_time = datetime.now()
        
        context_parts = []
        references = []
        max_context_length = 6000  
        
        for i, result in enumerate(retrieved_chunks):
            chunk = result.chunk
            if chunk.document_type == "pdf":
                page_num = chunk.metadata.get("page_number", "N/A")
                line_range = chunk.metadata.get("line_range", (chunk.line_number, chunk.line_number))
                chunk_text = f"[Source {i+1}: {chunk.source_file}, Page {page_num}, Lines {line_range[0]}-{line_range[1]}, Clause ID: {chunk.clause_id or 'N/A'}]\n{chunk.content}"
            elif chunk.document_type == "docx":
                para_num = chunk.metadata.get("paragraph_number", chunk.line_number)
                chunk_text = f"[Source {i+1}: {chunk.source_file}, Paragraph {para_num}, Clause ID: {chunk.clause_id or 'N/A'}]\n{chunk.content}"
            else:
                chunk_text = f"[Source {i+1}: {chunk.source_file}, Line {chunk.line_number}, Clause ID: {chunk.clause_id or 'N/A'}]\n{chunk.content}"
            if len(''.join(context_parts)) + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            references.append({
                "source": chunk.source_file,
                "page_number": chunk.metadata.get("page_number", None),
                "line_number": chunk.line_number,  # Start line
                "line_range": chunk.metadata.get("line_range", (chunk.line_number, chunk.line_number)),
                "clause_id": chunk.clause_id,
                "clause": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                "relevance_score": result.relevance_score,
                "weight": result.weight
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert legal document analyst. Provide a concise, accurate answer to the query based only on the provided context. If the query asks for specific details such as durations, locations, or conditions, explicitly extract and include those details from the clauses (e.g., exact duration like '30 days' or location like 'New York'). Include suggestions only if the query explicitly asks for them. Return a structured JSON response with the answer, suggestions (if applicable), detailed references to specific clauses including document name, page number (if applicable), line numbers, and clause ID (e.g., 'Clause a.i.2'), and a clear rationale.

Query: {query}

Context: {context}

Return a JSON object in this format:
{{
    "response": "Concise answer with specific details if applicable",
    "suggestions": ["Suggestion 1"] | [],
    "references": [
        {{"source": "file_path", "page_number": 1, "line_number": 10, "line_range": [10, 15], "clause_id": "Clause a.i.2", "clause": "Clause text snippet"}}
    ],
    "rationale": "Explanation citing specific clauses with clause IDs",
    "confidence_score": 0.95
}}

Guidelines:
- Answer must be concise and directly address the query.
- Only use information from the context; do not hallucinate.
- Extract and state specific durations (e.g., '6 months') or locations (e.g., 'London') if mentioned in the query and present in the context.
- Include detailed references with document name, page number (for PDFs), line range, and clause ID.
- Cite clauses using their clause_id (e.g., 'Section 1.1(a)') in the rationale and references.
- Assign confidence score (0.0-1.0) based on information clarity and completeness.
- If information is insufficient, state so explicitly in the response and rationale.

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                top_p=0.9
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                llm_result = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                else:
                    llm_result = {
                        "response": "Error: Could not parse LLM response",
                        "suggestions": [],
                        "references": [],
                        "rationale": "Invalid JSON response from LLM",
                        "confidence_score": 0.1
                    }
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return QueryResponse(
                query=query,
                answer={
                    "response": llm_result.get("response", "No answer provided"),
                    "suggestions": llm_result.get("suggestions", []),
                    "references": llm_result.get("references", []),
                    "rationale": llm_result.get("rationale", "No rationale provided"),
                    "confidence_score": float(llm_result.get("confidence_score", 0.5))
                },
                confidence_score=float(llm_result.get("confidence_score", 0.5)),
                decision_rationale=llm_result.get("rationale", "No rationale provided"),
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )
        
        except Exception as e:
            logger.error(f"Error generating Groq LLM response: {e}")
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return QueryResponse(
                query=query,
                answer={
                    "response": "Error processing query",
                    "suggestions": [],
                    "references": references,
                    "rationale": f"Error: {str(e)}",
                    "confidence_score": 0.0
                },
                confidence_score=0.0,
                decision_rationale=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )

class IntelligentQuerySystem:
    """Orchestrates document processing, retrieval, and response generation."""
    
    def __init__(self, groq_api_key: str, cache_dir: str = "./cache"):
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabase()
        self.llm_processor = LLMProcessor(groq_api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    async def process_documents_async(self, file_paths: List[str]) -> List[DocumentChunk]:
        """Process multiple documents asynchronously."""
        all_chunks = []
        for file_path in file_paths:
            logger.info(f"Processing document: {file_path}")
            try:
                chunks = self.document_processor.process_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    async def build_knowledge_base(self, file_paths: List[str], save_index: bool = True):
        """Build the knowledge base from legal documents."""
        chunks = await self.process_documents_async(file_paths)
        if not chunks:
            logger.error("No document chunks created. Check file paths and contents.")
            raise ValueError("No document chunks to index.")
        self.vector_db.build_index(chunks)
        if save_index:
            index_path = self.cache_dir / "vector_index"
            self.vector_db.save_index(str(index_path))
    
    def load_knowledge_base(self, index_path: str = None):
        """Load pre-built knowledge base."""
        if index_path is None:
            index_path = str(self.cache_dir / "vector_index")
        self.vector_db.load_index(index_path)
        logger.info("Knowledge base loaded successfully")
    
    def query(self, query_text: str) -> QueryResponse:
        """Process a query and return structured JSON response."""
        logger.info(f"Processing query: {query_text}")
        retrieved_results = self.vector_db.search(query_text, k=5)
        response = self.llm_processor.generate_response(query_text, retrieved_results)
        return response
    
    async def batch_query(self, queries: List[str]) -> List[QueryResponse]:
        """Process multiple queries in batch."""
        responses = []
        for query in queries:
            response = self.query(query)
            responses.append(response)
        return responses

async def main():
    """Usage of the Intelligent Query System."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        return
    
    system = IntelligentQuerySystem(groq_api_key=groq_api_key, cache_dir="./query_system_cache")
    
    # Scan the Documents folder for supported files
    documents_dir = Path("Documents")
    supported_exts = {'.pdf', '.docx', '.eml', '.txt'}
    document_paths = [
        str(path) for path in documents_dir.rglob("*")
        if path.suffix.lower() in supported_exts
    ]
    
    valid_paths = [path for path in document_paths if os.path.exists(path)]
    if not valid_paths:
        print("No valid document paths provided. Please check file paths.")
        return
    
    print("Building knowledge base...")
    await system.build_knowledge_base(valid_paths)
    
    sample_queries = [
    "Does the policy cover cataract operation and what is the limit if the sum insured is ₹1,00,000?",
    "Does the policy cover 23-hour oral chemotherapy under day care treatment?",
    "Is a 26-year-old unemployed dependent child eligible for coverage?",
    "Will cataract surgery in the 11th month of the first policy year be covered?",
    "Is robotic surgery for prostate vaporization fully covered in a PPN hospital?",
    "Will cleft palate repair surgery be covered under cosmetic or medically necessary clause?",
    "Can steam inhaler used at home post-discharge be reimbursed under post-hospitalization expenses?",
    "Will the policy cover 3-day inpatient Ayurvedic treatment in a co-located private AYUSH center?",
    "Are surgical gloves and ECG electrodes used during ICU heart surgery reimbursable?",
    "What happens to the waiting period for pre-existing diseases if sum insured is increased after 2 years?",
    "Will a pre-existing illness declared after 60 months of coverage be rejected?",
    "How is cumulative bonus adjusted after 4 claim-free years followed by a claim in year 5?",
    "What documents are required if TPA denies cashless and patient is treated in a non-network hospital?",
    "What is the arbitration process if insurer disputes claim amount but not liability?",
    "Who receives claim payment if insured dies after discharge but before claim submission?"
]
    
    print("\nProcessing queries...")
    responses = await system.batch_query(sample_queries)
    
    results = {
        "llm_provider": "Groq",
        "model_used": system.llm_processor.model,
        "queries_processed": len(responses),
        "timestamp": datetime.now().isoformat(),
        "responses": [asdict(response) for response in responses]
    }
    
    with open("query_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nQuery results:")
    for response in responses:
        print(f"\nQuery: {response.query}")
        print(f"Response: {response.answer['response']}")
        if response.answer['suggestions']:
            print(f"Suggestions: {response.answer['suggestions']}")
        print(f"References: {len(response.answer['references'])} clauses")
        print(f"Rationale: {response.answer['rationale'][:100]}...")
        print(f"Confidence: {response.confidence_score}")
        print(f"Processing time: {response.processing_time_ms}ms")
        print("-" * 80)
    
    print(f"\nResults saved to query_results.json")

if __name__ == "__main__":
    asyncio.run(main())