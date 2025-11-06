import os
from typing import Dict, List

class GroqConfig:
    """Configuration for Groq models"""
    
    # Available Groq models with their specifications
    MODELS = {
        "llama-3.3-70b-versatile": {
            "name": "llama-3.3-70b-versatile",
            "max_tokens": 8192,
            "context_window": 131072,
            "description": "High-quality, versatile model for complex tasks"
        },
        "llama-3.1-8b-instant": {
            "name": "llama-3.1-8b-instant", 
            "max_tokens": 8192,
            "context_window": 131072,
            "description": "Fast, efficient model for simpler tasks"
        },
        "mixtral-8x7b-32768": {
            "name": "mixtral-8x7b-32768",
            "max_tokens": 4096,
            "context_window": 32768,
            "description": "Mixture of Experts model for balanced performance"
        }
    }
    
    # Model selection based on task type
    TASK_MODELS = {
        "qa": "llama-3.1-8b-instant",  # Fast for Q&A
        "summary": "llama-3.3-70b-versatile",  # High quality for summaries
        "extraction": "llama-3.3-70b-versatile",  # High quality for structured data
        "routing": "llama-3.1-8b-instant"  # Fast for routing decisions
    }
    
    @classmethod
    def get_model_for_task(cls, task_type: str) -> str:
        return cls.TASK_MODELS.get(task_type, "llama-3.1-8b-instant")