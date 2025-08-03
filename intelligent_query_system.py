import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import hashlib
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

# LLM integration using Groq
from groq import Groq

# Additional utilities
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    source_file: str
    chunk_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    document_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with relevance score."""
    chunk: DocumentChunk
    relevance_score: float
    embedding_similarity: float

@dataclass
class QueryResponse:
    """Structured response format for query results."""
    query: str
    answer: str
    confidence_score: float
    supporting_evidence: List[Dict[str, Any]]
    decision_rationale: str
    relevant_clauses: List[str]
    conditions_and_exclusions: List[str]
    recommendations: List[str]
    timestamp: str
    processing_time_ms: int

class DocumentProcessor:
    """Handles processing of various document formats."""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.eml', '.txt'}
    
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from PDF files."""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunk_id = f"{Path(file_path).stem}_page_{page_num}"
                        chunks.append(DocumentChunk(
                            content=text,
                            source_file=file_path,
                            chunk_id=chunk_id,
                            page_number=page_num + 1,
                            document_type="pdf",
                            metadata={"total_pages": len(reader.pages)}
                        ))
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
        return chunks
    
    def process_docx(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from DOCX files."""
        chunks = []
        try:
            doc = docx.Document(file_path)
            full_text = []
            current_section = None
            
            for para in doc.paragraphs:
                if para.text.strip():
                    # Detect section headers (bold text or specific patterns)
                    if para.runs and para.runs[0].bold:
                        current_section = para.text.strip()
                    full_text.append(para.text)
            
            # Create chunks based on paragraphs or sections
            content = '\n'.join(full_text)
            chunk_id = f"{Path(file_path).stem}_full"
            chunks.append(DocumentChunk(
                content=content,
                source_file=file_path,
                chunk_id=chunk_id,
                section_title=current_section,
                document_type="docx",
                metadata={"paragraph_count": len(doc.paragraphs)}
            ))
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
        return chunks
    
    def process_email(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from email files."""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)
                
                # Extract email metadata
                subject = msg.get('Subject', 'No Subject')
                sender = msg.get('From', 'Unknown Sender')
                date = msg.get('Date', 'Unknown Date')
                
                # Extract body content
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body += part.get_content()
                else:
                    body = msg.get_content()
                
                chunk_id = f"{Path(file_path).stem}_email"
                chunks.append(DocumentChunk(
                    content=f"Subject: {subject}\n\n{body}",
                    source_file=file_path,
                    chunk_id=chunk_id,
                    document_type="email",
                    metadata={
                        "sender": sender,
                        "subject": subject,
                        "date": date
                    }
                ))
        except Exception as e:
            logger.error(f"Error processing email {file_path}: {e}")
        return chunks
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document based on its file extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.process_pdf(file_path)
        elif file_ext == '.docx':
            return self.process_docx(file_path)
        elif file_ext == '.eml':
            return self.process_email(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [DocumentChunk(
                    content=content,
                    source_file=file_path,
                    chunk_id=f"{Path(file_path).stem}_txt",
                    document_type="text"
                )]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

class TextChunker:
    """Handles intelligent text chunking for better retrieval."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stop_words = set(stopwords.words('english'))
    
    def chunk_by_sentences(self, text: str, source_file: str, base_chunk: DocumentChunk) -> List[DocumentChunk]:
        """Create chunks based on sentence boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_counter = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_id = f"{base_chunk.chunk_id}_sub_{chunk_counter}"
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    source_file=source_file,
                    chunk_id=chunk_id,
                    page_number=base_chunk.page_number,
                    section_title=base_chunk.section_title,
                    document_type=base_chunk.document_type,
                    metadata=base_chunk.metadata
                ))
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk.split()[-self.overlap:])
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk.split())
                chunk_counter += 1
            else:
                current_chunk += " " + sentence
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"{base_chunk.chunk_id}_sub_{chunk_counter}"
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                source_file=source_file,
                chunk_id=chunk_id,
                page_number=base_chunk.page_number,
                section_title=base_chunk.section_title,
                document_type=base_chunk.document_type,
                metadata=base_chunk.metadata
            ))
        
        return chunks

class VectorDatabase:
    """Manages vector embeddings and similarity search using FAISS."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for document chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index from document chunks."""
        logger.info(f"Building vector index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        self.embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info("Vector index built successfully")
    
    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Search for relevant chunks using semantic similarity."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    relevance_score=float(score),
                    embedding_similarity=float(score)
                ))
        
        return results
    
    def save_index(self, path: str):
        """Save the FAISS index and metadata to disk."""
        os.makedirs(path, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(os.path.join(path, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load the FAISS index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load chunks and embeddings
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        with open(os.path.join(path, "embeddings.pkl"), "rb") as f:
            self.embeddings = pickle.load(f)
        
        logger.info(f"Index loaded from {path}")

class LLMProcessor:
    """Handles LLM interactions for query processing and response generation using Groq."""
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base") 
        
        # Model-specific configurations
        self.model_configs = {
            "llama3-70b-8192": {"max_tokens": 8192, "context_window": 8192},
            "llama3-8b-8192": {"max_tokens": 8192, "context_window": 8192},
            "mixtral-8x7b-32768": {"max_tokens": 32768, "context_window": 32768},
            "gemma-7b-it": {"max_tokens": 8192, "context_window": 8192}
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate)."""
        return len(self.tokenizer.encode(text))
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for current model."""
        return self.model_configs.get(self.model, {"max_tokens": 8192})["max_tokens"]
    
    def extract_clauses(self, content: str, query: str) -> List[str]:
        """Extract relevant clauses from content using Groq LLM."""
        # Truncate content if too long
        max_content_tokens = min(4000, self.get_max_tokens() - 1000)
        if self.count_tokens(content) > max_content_tokens:
            content = content[:max_content_tokens * 4]  # Rough character estimate
        
        prompt = f"""You are an expert document analyst. Extract specific clauses from the following content that directly relate to the query.

Query: {query}

Content: {content}

Extract only the most relevant clauses that directly address the query. Return your response as a JSON object with a "clauses" array containing the extracted clauses as strings.

Example format:
{{"clauses": ["First relevant clause here", "Second relevant clause here"]}}

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
            
            # Try to parse JSON, with fallback handling
            try:
                result = json.loads(content)
                return result.get("clauses", [])
            except json.JSONDecodeError:
                # Try to extract JSON from response if wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result.get("clauses", [])
                else:
                    logger.warning("Could not parse clause extraction response as JSON")
                    return []
                    
        except Exception as e:
            logger.error(f"Error extracting clauses: {e}")
            return []
    
    def generate_response(self, query: str, retrieved_chunks: List[RetrievalResult]) -> QueryResponse:
        """Generate comprehensive response using Groq LLM."""
        start_time = datetime.now()
        
        # Prepare context from retrieved chunks
        context_parts = []
        supporting_evidence = []
        
        # Calculate available tokens for context
        max_context_tokens = min(6000, self.get_max_tokens() - 2000)  # Reserve tokens for prompt and response
        current_tokens = 0
        
        for i, result in enumerate(retrieved_chunks):
            chunk = result.chunk
            chunk_text = f"[Source {i+1}: {chunk.source_file} - {chunk.chunk_id}]\n{chunk.content}"
            chunk_tokens = self.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > max_context_tokens:
                break
                
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
            
            supporting_evidence.append({
                "source": chunk.source_file,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "relevance_score": result.relevance_score,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title
            })
        
        context = "\n\n".join(context_parts)
        
        # Create comprehensive prompt optimized for Groq models
        prompt = f"""You are an expert document analyst specializing in insurance, legal, HR, and compliance domains. Analyze the provided context and answer the user's query with precision and clarity.

Query: {query}

Context from retrieved documents:
{context}

Provide a comprehensive analysis in the following JSON format. Ensure your response is valid JSON:

{{
    "answer": "Direct, clear answer to the query based on the provided context",
    "confidence_score": 0.95,
    "decision_rationale": "Detailed explanation of how you arrived at this answer, citing specific sources",
    "relevant_clauses": ["Direct quote from clause 1", "Direct quote from clause 2"],
    "conditions_and_exclusions": ["Specific condition 1", "Specific exclusion 1"],
    "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2"]
}}

Guidelines:
1. Base your answer ONLY on the provided context
2. Be specific and cite exact clauses when possible
3. Clearly distinguish between coverage, conditions, and exclusions
4. Provide actionable recommendations
5. Assign confidence score (0.0-1.0) based on information clarity and completeness
6. If information is insufficient, clearly state what additional details are needed
7. Ensure all JSON values are properly quoted strings or numbers

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=min(2000, self.get_max_tokens() - self.count_tokens(prompt)),
                top_p=0.9
            )
            
            # Parse LLM response with robust error handling
            content = response.choices[0].message.content.strip()
            
            try:
                llm_result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response if wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                else:
                    # Fallback: create structured response from text
                    logger.warning("Could not parse LLM response as JSON, creating fallback response")
                    llm_result = {
                        "answer": content[:500] if content else "No response generated",
                        "confidence_score": 0.3,
                        "decision_rationale": "Response could not be parsed as structured JSON",
                        "relevant_clauses": [],
                        "conditions_and_exclusions": [],
                        "recommendations": ["Please retry the query"]
                    }
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return QueryResponse(
                query=query,
                answer=llm_result.get("answer", "No answer provided"),
                confidence_score=float(llm_result.get("confidence_score", 0.5)),
                supporting_evidence=supporting_evidence,
                decision_rationale=llm_result.get("decision_rationale", "No rationale provided"),
                relevant_clauses=llm_result.get("relevant_clauses", []),
                conditions_and_exclusions=llm_result.get("conditions_and_exclusions", []),
                recommendations=llm_result.get("recommendations", []),
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating Groq LLM response: {e}")
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return QueryResponse(
                query=query,
                answer="Error processing query with Groq LLM. Please try again.",
                confidence_score=0.0,
                supporting_evidence=supporting_evidence,
                decision_rationale=f"Groq processing error: {str(e)}",
                relevant_clauses=[],
                conditions_and_exclusions=[],
                recommendations=["Please retry the query or contact support"],
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )

class IntelligentQuerySystem:
    """Main system orchestrating document processing, retrieval, and response generation."""
    
    def __init__(self, groq_api_key: str, cache_dir: str = "./cache", model: str = "llama3-70b-8192"):
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        self.vector_db = VectorDatabase()
        self.llm_processor = LLMProcessor(groq_api_key, model)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Domain-specific configurations
        self.domain_configs = {
            "insurance": {
                "keywords": ["coverage", "premium", "deductible", "claim", "exclusion", "policy"],
                "chunk_size": 800,
                "retrieval_k": 8
            },
            "legal": {
                "keywords": ["contract", "clause", "terms", "conditions", "liability", "breach"],
                "chunk_size": 1200,
                "retrieval_k": 6
            },
            "hr": {
                "keywords": ["employee", "benefits", "policy", "procedure", "compliance", "handbook"],
                "chunk_size": 600,
                "retrieval_k": 10
            },
            "compliance": {
                "keywords": ["regulation", "standard", "requirement", "audit", "risk", "control"],
                "chunk_size": 1000,
                "retrieval_k": 7
            }
        }
    
    def detect_domain(self, query: str) -> str:
        """Detect the domain of the query for optimized processing."""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, config in self.domain_configs.items():
            score = sum(1 for keyword in config["keywords"] if keyword in query_lower)
            domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"
    
    async def process_documents_async(self, file_paths: List[str]) -> List[DocumentChunk]:
        """Process multiple documents asynchronously."""
        all_chunks = []
        
        for file_path in file_paths:
            logger.info(f"Processing document: {file_path}")
            
            try:
                # Process document
                base_chunks = self.document_processor.process_document(file_path)
                
                # Further chunk large documents
                for base_chunk in base_chunks:
                    if len(base_chunk.content.split()) > 1000:
                        sub_chunks = self.text_chunker.chunk_by_sentences(
                            base_chunk.content, file_path, base_chunk
                        )
                        all_chunks.extend(sub_chunks)
                    else:
                        all_chunks.append(base_chunk)
                        
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def build_knowledge_base(self, file_paths: List[str], save_index: bool = True):
        """Build the knowledge base from documents."""
        # Process documents
        chunks = asyncio.run(self.process_documents_async(file_paths))
        
        if not chunks:
            logger.error("No document chunks were created. Please check your file paths and document contents.")
            raise ValueError("No document chunks to index. Aborting knowledge base build.")
        
        # Build vector index
        self.vector_db.build_index(chunks)
        
        # Save index if requested
        if save_index:
            index_path = self.cache_dir / "vector_index"
            self.vector_db.save_index(str(index_path))
    
    def load_knowledge_base(self, index_path: str = None):
        """Load pre-built knowledge base."""
        if index_path is None:
            index_path = str(self.cache_dir / "vector_index")
        
        self.vector_db.load_index(index_path)
        logger.info("Knowledge base loaded successfully")
    
    def query(self, query_text: str, domain: str = None) -> QueryResponse:
        """Process a query and return structured response."""
        if domain is None:
            domain = self.detect_domain(query_text)
            self.domain_configs = {
            "insurance": {
                "keywords": ["coverage", "premium", "deductible", "claim", "exclusion", "policy"],
                "chunk_size": 800,
                "retrieval_k": 8
            },
            "legal": {
                "keywords": ["contract", "clause", "terms", "conditions", "liability", "breach"],
                "chunk_size": 1200,
                "retrieval_k": 6
            },
            "hr": {
                "keywords": ["employee", "benefits", "policy", "procedure", "compliance", "handbook"],
                "chunk_size": 600,
                "retrieval_k": 10
            },
            "compliance": {
                "keywords": ["regulation", "standard", "requirement", "audit", "risk", "control"],
                "chunk_size": 1000,
                "retrieval_k": 7
            },
            "general": {
                "keywords": [],
                "chunk_size": 1000,
                "retrieval_k": 8
            }
        }
        
        logger.info(f"Processing query in {domain} domain: {query_text}")
        config = self.domain_configs.get(domain, self.domain_configs["general"])
        k = config.get("retrieval_k", 8)
        retrieved_results = self.vector_db.search(query_text, k=k)
        response = self.llm_processor.generate_response(query_text, retrieved_results)
        
        return response
    
    def batch_query(self, queries: List[str]) -> List[QueryResponse]:
        """Process multiple queries in batch."""
        responses = []
        for query in queries:
            response = self.query(query)
            responses.append(response)
        return responses


def main():
    """Usage of the Intelligent Query System with Groq."""
    system = IntelligentQuerySystem(
        groq_api_key="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",
        cache_dir="./query_system_cache",
        model="llama3-70b-8192" 
    )
    
    
    document_paths = [
        "Documents/policy1.pdf",
        #  "documents/contract1.docx"
    ]
    print("Building knowledge base...")
    system.build_knowledge_base(document_paths)
    
    sample_queries = [
        "Does this policy cover knee surgery, and what are the conditions?",
        "What is the process for filing a workers compensation claim?",
        "What are the data retention requirements for customer information?",
        "What benefits are available for remote employees?"
    ]
    
    print("\nProcessing queries with Groq LLM...")
    for query in sample_queries:
        response = system.query(query)
        
        print(f"\nQuery: {query}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score}")
        print(f"Processing time: {response.processing_time_ms}ms")
        print(f"Supporting evidence sources: {len(response.supporting_evidence)}")
        if response.relevant_clauses:
            print(f"Relevant clauses found: {len(response.relevant_clauses)}")
        print("-" * 80)
    
    batch_responses = system.batch_query(sample_queries)
    results = {
        "llm_provider": "Groq",
        "model_used": system.llm_processor.model,
        "queries_processed": len(batch_responses),
        "timestamp": datetime.now().isoformat(),
        "responses": [asdict(response) for response in batch_responses]
    }
    
    with open("groq_query_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBatch processing complete. Results saved to groq_query_results.json")
    print(f"Using Groq model: {system.llm_processor.model}")

if __name__ == "__main__":
    main()