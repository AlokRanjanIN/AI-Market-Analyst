from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

class DocumentProcessor:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.embeddings = None
        
    def load_embeddings(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load embedding model with comparison capability"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return self.embeddings
    
    def chunk_document(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Split document into chunks with overlap"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
        return documents
    
    def create_vector_store(self, documents: list):
        """Create and persist vector store"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()
        return self.vector_store
    
    def load_vector_store(self):
        """Load existing vector store"""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if self.vector_store is None:
            self.load_vector_store()
        return self.vector_store.similarity_search(query, k=k)