# AI Market Analyst Agent with Groq

A multi-functional AI agent that processes market research documents using Groq's high-performance LLMs.

## Features

- **General Q&A**: Answer specific questions using llama-3.1-8b-instant for fast responses
- **Research Summary**: Generate comprehensive summaries using llama-3.3-70b-versatile for high quality
- **Structured Extraction**: Extract market data and SWOT analysis using high-quality models
- **Autonomous Routing**: AI-powered query routing to appropriate task
- **Streamlit UI**: User-friendly web interface
- **Docker Support**: Containerized deployment

## Setup & Installation

### Prerequisites
- Python 3.9+
- Groq API key
- Docker (optional)

### Local Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-market-analyst
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables**
```bash
export GROQ_API_KEY="your-groq-api-key"
```

4. **Run the application**
```bash
# Start the API server
uvicorn app.main:app --reload --port 8000

# In another terminal, start the Streamlit UI
streamlit run ui/streamlit_app.py
```

### Docker Installation

1. **Build and run with Docker**
```bash
docker build -t ai-market-analyst .
docker run -p 8000:8000 -e GROQ_API_KEY="your-groq-key" ai-market-analyst
```

## Groq Model Configuration

The system uses different Groq models optimized for specific tasks:

| Task | Model | Reason |
|------|-------|--------|
| Q&A | `llama-3.1-8b-instant` | Fast response times for interactive queries |
| Summary | `llama-3.3-70b-versatile` | High-quality analysis for comprehensive summaries |
| Extraction | `llama-3.3-70b-versatile` | Accurate structured data extraction |
| Routing | `llama-3.1-8b-instant` | Quick decision making for query routing |

## API Usage Examples

### General Q&A
```bash
curl -X POST "http://localhost:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Innovate Inc. market share?"}'
```

### Autonomous Analysis
```bash
curl -X POST "http://localhost:8000/autonomous" \
     -H "Content-Type: application/json" \
     -d '{"query": "Summarize the competitive landscape and extract market share data"}'
```

## Design Decisions (Updated for Groq)

### Model Selection Strategy
**Multi-Model Approach**: Using different models for different tasks
- **llama-3.1-8b-instant**: For Q&A and routing (low latency, good enough quality)
- **llama-3.3-70b-versatile**: For summarization and extraction (high quality, more context)

**Justification**:
- **Cost Efficiency**: Smaller models for simpler tasks reduce API costs
- **Performance**: Faster responses for interactive Q&A
- **Quality**: Larger models for complex analysis tasks
- **Groq Advantage**: Extremely fast inference speeds compared to traditional APIs

### Prompt Optimization for Groq
- **Structured JSON Output**: Enhanced prompts with explicit formatting requirements
- **Error Handling**: Robust fallback mechanisms for parsing responses
- **Context Management**: Optimized context windows for each model type

## Performance Benefits with Groq

- **Response Times**: 2-5x faster than traditional APIs
- **Throughput**: Higher requests per minute limits
- **Cost**: More efficient pricing for high-volume usage
- **Latency**: Sub-second responses for most queries

## Available Groq Models

The system can be easily configured to use any of these Groq models:
- `llama-3.3-70b-versatile` (Primary for complex tasks)
- `llama-3.1-8b-instant` (Primary for fast tasks) 
- `mixtral-8x7b-32768` (Alternative for balanced performance)
- `gemma2-9b-it` (Lightweight alternative)

View available models: `GET http://localhost:8000/models`

## Key Advantages of Using Groq

1. **Extreme Speed**: Groq provides significantly faster inference times
2. **High Throughput**: Better rate limits for production usage
3. **Cost Effective**: More affordable for high-volume applications
4. **Model Variety**: Multiple model options for different use cases
5. **No Latency**: Sub-second responses for better user experience

The solution leverages Groq's strengths by:
- Using faster models (llama-3.1-8b-instant) for simple Q&A
- Using higher-quality models (llama-3.3-70b-versatile) for complex analysis
- Implementing robust error handling for API reliability
- Maintaining all the original functionality with improved performance