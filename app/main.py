from fastapi import FastAPI, HTTPException
from .models import (
    QARequest, QAResponse, ResearchSummaryRequest, 
    ResearchSummaryResponse, ExtractionRequest, ExtractionResponse,
    AutonomousRequest, AutonomousResponse
)
from .agent import AIMarketAnalyst
from .document_processor import DocumentProcessor
import os
import asyncio

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="AI Market Analyst - Groq", version="1.0.0")

# Global agent instance
analyst_agent = None
init_task = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent on startup"""
    global analyst_agent
    
    # Initialize agent with Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    # Do heavy initialization in executor
    loop = asyncio.get_running_loop()
    dp = await loop.run_in_executor(None, init_document_processor, groq_api_key)
    

    analyst_agent = AIMarketAnalyst(
        document_processor=dp,
        groq_api_key=groq_api_key
    )

def init_document_processor(groq_api_key):
    dp = DocumentProcessor()
    dp.load_embeddings()
    if dp.persist_directory.exists():
        dp.load_vector_store()
    else:
        with open("app/data/market_research.txt", "r") as f:
            document_text = f.read()
        documents = dp.chunk_document(document_text)
        dp.create_vector_store(documents)
    return dp

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

@app.post("/qa", response_model=QAResponse)
async def general_qa(request: QARequest):
    """General Q&A endpoint"""
    try:
        if analyst_agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        result = analyst_agent.general_qa(request.question)
        return QAResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary", response_model=ResearchSummaryResponse)
async def research_summary(request: ResearchSummaryRequest = None):
    """Market research summary endpoint"""
    try:
        if analyst_agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        focus_areas = request.focus_areas if request else None
        # print(f"focus_areas: {focus_areas}")
        result = analyst_agent.market_research_summary(focus_areas)
        # print(f"result: {result}")
        return ResearchSummaryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract", response_model=ExtractionResponse)
async def structured_extraction(request: ExtractionRequest = None):
    """Structured data extraction endpoint"""
    try:
        if analyst_agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        entities = request.entities if request else None
        result = analyst_agent.structured_data_extraction(entities)
        return ExtractionResponse(extracted_data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autonomous", response_model=AutonomousResponse)
async def autonomous_analysis(request: AutonomousRequest):
    """Autonomous routing endpoint"""
    try:
        if analyst_agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        result = analyst_agent.autonomous_router(request.query)
        # print("DEBUG Autonomous Result:", result)
        return AutonomousResponse(**result)
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": analyst_agent is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)