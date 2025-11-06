FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data chroma_db

# Copy the market research data
COPY data/market_research.txt data/

EXPOSE 8000

# Use environment variable for Groq API key
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]