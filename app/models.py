from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    sources: List[str]

class ResearchSummaryRequest(BaseModel):
    focus_areas: Optional[List[str]] = None

class ResearchSummaryResponse(BaseModel):
    summary: str
    key_findings: List[str]
    recommendations: List[str]

class ExtractionRequest(BaseModel):
    entities: List[str]

class ExtractionResponse(BaseModel):
    extracted_data: Dict[str, Any]

class MarketData(BaseModel):
    market_size: str
    growth_rate: str
    market_share: Dict[str, str]
    competitors: List[str]
    swot_analysis: Dict[str, List[str]]

class AutonomousRequest(BaseModel):
    query: str

class AutonomousResponse(BaseModel):
    task_type: str
    response: Any