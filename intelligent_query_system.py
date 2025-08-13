import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from pathlib import Path
import PyPDF2
import faiss
import numpy as np
import pickle
from groq import Groq
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import re

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
    """Handles processing of legal documents (PDF only)."""
    
    def __init__(self):
        self.supported_formats = {'.pdf'}
    
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
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document based on its file extension (only PDF)."""
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_ext}")
            return []
        return self.process_pdf(file_path)

class VectorDatabase:
    """Manages vector embeddings and semantic search using FAISS."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.legal_keywords = [
            "clause", "section", "article", "provision", "term", "condition", 
            "disclaimer", "disclosure", "policy", "contract", "agreement", 
            "liability", "obligation", "warranty", "indemnity"
        ]
    
    def ensure_embedding_model_loaded(self):
        """Lazily load the SentenceTransformer to avoid slow imports at server startup."""
        if self.embedding_model is None:
            # Local import to defer heavy dependency loading until needed
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for document chunks."""
        self.ensure_embedding_model_loaded()
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
        
        self.ensure_embedding_model_loaded()
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
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.client = Groq(api_key=api_key)
        # Prefer env override, then provided param, then a widely available Groq model
        self.model = os.getenv("GROQ_MODEL", model or "llama3-8b-8192")
        self.max_tokens = 8192

    @staticmethod
    def _repair_and_parse_json(raw_text: str) -> Dict[str, Any]:
        """Best-effort JSON extraction/repair for occasionally-invalid LLM outputs."""
        # 1) Try direct parse
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        # 2) Extract the first JSON object-looking block
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            candidate = match.group(0)
            # Normalize smart quotes
            candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
            # Escape stray backslashes not part of a valid JSON escape
            candidate = re.sub(r"(?<!\\)\\(?![\\\"/bfnrtu])", r"\\\\", candidate)
            # Remove trailing commas before closing } or ]
            candidate = re.sub(r",\s*([}\]])", r"\\1", candidate)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # 3) Fallback minimal structure
        return {
            "response": "Unable to parse model output into structured JSON.",
            "suggestions": [],
            "references": [],
            "rationale": "Invalid JSON returned by model",
            "confidence_score": 0.1,
        }
    
    def generate_response(self, query: str, retrieved_chunks: List[RetrievalResult]) -> QueryResponse:
        """Generate structured JSON response with detailed references and specific details."""
        start_time = datetime.now()
        
        context_parts = []
        references = []
        max_context_length = 6000  
        
        for i, result in enumerate(retrieved_chunks):
            chunk = result.chunk
            page_num = chunk.metadata.get("page_number", "N/A")
            line_range = chunk.metadata.get("line_range", (chunk.line_number, chunk.line_number))
            chunk_text = f"[Source {i+1}: {chunk.source_file}, Page {page_num}, Lines {line_range[0]}-{line_range[1]}, Clause ID: {chunk.clause_id or 'N/A'}]\n{chunk.content}"
            if len(''.join(context_parts)) + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            references.append({
                "source": chunk.source_file,
                "page_number": page_num,
                "line_number": chunk.line_number,
                "line_range": line_range,
                "clause_id": chunk.clause_id or "N/A",
                "clause": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                "relevance_score": result.relevance_score,
                "weight": result.weight
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert legal document analyst and a friendly, knowledgeable assistant. Please provide a concise, accurate, and conversational answer to the query based solely on the provided context. If the query asks for specific details like durations, locations, or conditions, extract and include those details directly from the clauses (e.g., '30 days' or 'New York'). When answering, explicitly reference the specific clause, paragraph, page, section, and line numbers from the document where the information is found. Include suggestions only if the query explicitly requests them. If the query is unrelated to the content in the provided PDF, respond in a friendly, human-like way, explaining that it’s outside my knowledge base, such as: "I’m sorry, but that topic doesn’t seem to be covered in the document I have. I’d be happy to assist with something from the PDF if you’d like!" Return a structured JSON response with the answer, suggestions (if applicable), detailed references to specific clauses (including document name, page number, line numbers, and clause ID like 'Section 1.1(a)'), and a clear rationale.

Query: {query}

Context: {context}

Return a JSON object in this format:
{{
    "response": "Friendly, concise answer with specific details and references if applicable",
    "suggestions": ["Suggestion 1"] | [],
    "references": [
        {{"source": "file_path", "page_number": 1, "line_number": 10, "line_range": [10, 15], "clause_id": "Section 1.1(a)", "clause": "Clause text snippet"}}
    ],
    "rationale": "Clear explanation citing specific clauses with clause IDs",
    "confidence_score": 0.95
}}

Guidelines:
- Keep the answer concise, conversational, and directly tied to the query.
- Use only the provided context; do not make up information.
- Extract and mention specific details (e.g., '6 months' or 'London') if relevant and present in the context.
- Always include detailed references with document name, page number, line range, and clause ID in both the response and references field.
- Cite clauses using their clause_id (e.g., 'Section 1.1(a)') in the rationale and references.
- Assign a confidence score (0.0-1.0) based on the clarity and completeness of the information.
- If the information is insufficient, say so clearly in the response and rationale.
- For queries outside the context, provide a polite, human-like explanation.

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
            llm_result = self._repair_and_parse_json(content)
            
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
            # Friendly fallback when LLM call fails
            fallback_response = (
                "I’m sorry, I couldn’t complete the analysis due to an internal error. "
                "If you’re asking about something outside the uploaded documents, please upload a PDF that contains that information or try another query."
            )
            return QueryResponse(
                query=query,
                answer={
                    "response": fallback_response,
                    "suggestions": [],
                    "references": references,
                    "rationale": f"LLM error: {str(e)}",
                    "confidence_score": 0.0
                },
                confidence_score=0.0,
                decision_rationale=f"LLM error: {str(e)}",
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
    
    def query(self, query_text: str, k: int = 5) -> QueryResponse:
        """Process a query and return structured JSON response."""
        logger.info(f"Processing query: {query_text}")
        retrieved_results = self.vector_db.search(query_text, k=k)
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
    
    documents_dir = Path("Documents")
    supported_exts = {'.pdf'}
    document_paths = [
        str(path) for path in documents_dir.rglob("*")
        if path.suffix.lower() in supported_exts
    ]
    
    valid_paths = [path for path in document_paths if os.path.exists(path)]
    if not valid_paths:
        print("No valid PDF document paths provided. Please check file paths.")
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
        "Who receives claim payment if insured dies after discharge but before claim submission?",
        "What’s the weather like today in New York?"
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