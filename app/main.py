from fastapi import FastAPI, HTTPException
from .models import (
    QARequest, QAResponse, ResearchSummaryRequest, 
    ResearchSummaryResponse, ExtractionRequest, ExtractionResponse,
    AutonomousRequest, AutonomousResponse
)
from .agent import AIMarketAnalyst
from .document_processor import DocumentProcessor
import os

app = FastAPI(title="AI Market Analyst - Groq", version="1.0.0")

# Global agent instance
analyst_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent on startup"""
    global analyst_agent
    
    # Initialize document processor
    dp = DocumentProcessor()
    dp.load_embeddings()
    
    # Load or create vector store
    if os.path.exists("./chroma_db"):
        dp.load_vector_store()
    else:
        # Load document text
        with open("data/market_research.txt", "r") as f:
            document_text = f.read()
        
        # Process document
        documents = dp.chunk_document(document_text)
        dp.create_vector_store(documents)
    
    # Initialize agent with Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    analyst_agent = AIMarketAnalyst(
        document_processor=dp,
        groq_api_key=groq_api_key
    )

@app.get("/")
async def root():
    return {"message": "AI Market Analyst API with Groq is running"}

@app.get("/models")
async def list_models():
    """List available Groq models and configuration"""
    from .config import GroqConfig
    return {
        "available_models": GroqConfig.MODELS,
        "task_mapping": GroqConfig.TASK_MODELS
    }

# ... (keep all the existing endpoints the same)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)