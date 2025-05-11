import os
import logging
import re
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
import fitz  # PyMuPDF
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    
    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the Chroma collection."""
        super().__init__()
        self.file_path = file_path
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        self.vectorstore = None
        self.chunks = []  # Store chunks for inspection
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using PyMuPDF."""
        logger.info(f"Extracting text from PDF: {self.file_path}")
        try:
            doc = fitz.open(self.file_path)
            text = ""
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                text += page_text
                logger.info(f"Page {page_num}: Extracted {len(page_text)} characters")
            logger.info(f"Total pages: {len(doc)}, Total characters: {len(text)}")
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text using LangChain's SemanticChunker."""
        logger.info("Creating semantic chunks")
        try:
            chunker = SemanticChunker(
                embeddings=self.embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.7
            )
            chunks = chunker.split_text(raw_text)
            logger.info(f"Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:5], 1):  # Log first 5 chunks
                logger.info(f"Chunk {i}: {chunk[:200]}... (Length: {len(chunk)} chars)")
            return chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise

    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize the collection name to meet Chroma's requirements."""
        # Remove .pdf extension
        name = name.replace('.pdf', '')
        # Replace invalid characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Ensure starts and ends with alphanumeric
        name = name.strip('_')
        # Ensure length is 3-63 characters
        if len(name) < 3:
            name = name + 'doc'
        if len(name) > 63:
            name = name[:63].rstrip('_')
        return f"collection_{name}"

    def _process_document(self):
        """Process the document and add chunks to Chroma collection."""
        logger.info("Processing document")
        try:
            raw_text = self._extract_text()
            self.chunks = self._create_chunks(raw_text)
            
            docs = self.chunks
            metadatas = [{"source": os.path.basename(self.file_path), "chunk_id": i} for i in range(len(docs))]
            
            collection_name = self._sanitize_collection_name(os.path.basename(self.file_path))
            logger.info(f"Using collection name: {collection_name}")
            
            logger.info("Embedding chunks into Chroma")
            self.vectorstore = Chroma.from_texts(
                texts=docs,
                embedding=self.embedding_model,
                metadatas=metadatas,
                collection_name=collection_name
            )
            logger.info(f"Embedded {len(docs)} chunks into Chroma")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def _run(self, query: str, as_string: bool = True, k: int = 30) -> list | str:
        """Search the document with a query string. Returns list for debugging or string for agents."""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            structured_results = [{"content": result.page_content, "metadata": result.metadata} for result in results]
            if as_string:
                if not structured_results:
                    return "No results found"
                separator = "\n___\n"
                docs = [f"Chunk {i+1}: {result['content']}" for i, result in enumerate(structured_results)]
                return separator.join(docs)
            return structured_results
        except Exception as e:
            logger.error(f"Error running search: {str(e)}")
            return [] if not as_string else "Error during search"

    def inspect_chunks(self) -> list:
        """Return all chunks stored in the vector store."""
        try:
            collection = self.vectorstore._collection
            docs = collection.get(include=["documents", "metadatas"])
            return [{"content": doc, "metadata": meta} for doc, meta in zip(docs["documents"], docs["metadatas"])]
        except Exception as e:
            logger.error(f"Error inspecting chunks: {str(e)}")
            return []

# Test the implementation
def test_document_searcher():
    pdf_path = "/Users/monishkt/Downloads/ai-engineering-hub-main/agentic_rag/knowledge/MTD Goods tender.pdf"
    try:
        searcher = DocumentSearchTool(file_path=pdf_path)
        
        # Inspect chunks
        chunks = searcher.inspect_chunks()
        print(f"Total Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"Chunk {i}: {chunk['content'][:200]}... (Source: {chunk['metadata']['source']})")
        
        # Test search for ITB
        result = searcher._run("Instructions to Bidders", as_string=False)
        print("Search Results for 'Instructions to Bidders':")
        for i, res in enumerate(result):
            print(f"Result {i+1}: {res['content'][:200]}... (Source: {res['metadata']['source']})")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_document_searcher()