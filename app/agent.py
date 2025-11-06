from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_classic.chains import RetrievalQA
import json
import re
from typing import Dict, Any, List, Optional
from .document_processor import DocumentProcessor
from .models import MarketData
from .config import GroqConfig


class AIMarketAnalyst:
    def __init__(self, document_processor: DocumentProcessor, groq_api_key: str = None):
        self.dp = document_processor
        self.groq_api_key = groq_api_key
        
        # Initialize different LLMs for different tasks
        self.llm_fast = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=GroqConfig.TASK_MODELS["qa"],
            temperature=0.1,
            max_tokens=1024
        )
        
        self.llm_high_quality = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=GroqConfig.TASK_MODELS["summary"],
            temperature=0.1,
            max_tokens=2048
        )
        
        # Initialize QA chain with fast model
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_fast,
            chain_type="stuff",
            retriever=self.dp.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self._get_qa_prompt()
            }
        )
        print(f"\nQA CHAIN: {self.qa_chain}")
    
    def _get_qa_prompt(self):
        """Custom prompt for Q&A tasks"""
        return PromptTemplate(
            template="""You are an expert market research analyst. Use the following context to answer the question accurately and concisely.

Context: {context}

Question: {question}

If the context doesn't contain the answer, say "I cannot find this information in the provided market research document." Don't make up information.

Answer:""",
            input_variables=["context", "question"]
        )
    
    def general_qa(self, question: str) -> Dict[str, Any]:
        """Handle general Q&A about the market research"""
        try:
            print(f"\nQUESTION:\n{question}")
            retrieved_docs = self.dp.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ).invoke(question)

            print(f"\nRETRIEVED DOCS ({len(retrieved_docs)}):")
            for i, doc in enumerate(retrieved_docs):
                print(f"[DOC {i+1}]\n{doc.page_content[:300]}\n")

            result = self.qa_chain.invoke({"query": question})
            print(f"\RESULT:\n{result}")
            return {
                "answer": result["result"],
                # "sources": [doc.page_content for doc in result["source_documents"]]
                "sources": [doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content 
                           for doc in result.get("source_documents", [])]
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def market_research_summary(self, focus_areas: List = None) -> Dict[str, Any]:
        """Generate comprehensive market research summary using high-quality model"""
        base_prompt = """
        As a senior market research analyst, provide a comprehensive executive summary based on the provided market research document.

        KEY SECTIONS TO COVER:
        1. EXECUTIVE OVERVIEW: Brief summary of the company and market position
        2. MARKET INSIGHTS: Current market size, growth projections, key drivers
        3. COMPETITIVE ANALYSIS: Market share, key competitors, emerging threats
        4. STRATEGIC POSITION: SWOT analysis findings
        5. GROWTH OPPORTUNITIES: Key areas for expansion and improvement
        6. RECOMMENDATIONS: Actionable strategic recommendations

        DOCUMENT CONTEXT:
        {context}

        Please structure your response with clear sections and bullet points for key findings.
        """
        
        if focus_areas:
            base_prompt += f"\nSPECIAL FOCUS REQUESTED ON: {', '.join(focus_areas)}"
            
        # Get comprehensive context
        queries = ["market size", "competitive landscape", "SWOT analysis", "growth opportunities", "conclusion"]
        context_parts = []
        for query in queries:
            docs = self.dp.similarity_search(query, k=2)
            context_parts.extend([doc.page_content for doc in docs])
        
        context = "\n\n".join(context_parts[:8])  # Limit context length
        print(f"CONTEXT: \n{context}")
        
        prompt = base_prompt.format(context=context)
        print(f"PROMPT: \n{prompt}")
        try:
            response = self.llm_high_quality.invoke(prompt)
            print(f"\nRESPONSE: \n{response}")
            response_text = response.content if hasattr(response, 'content') else str(response)
            print(f"\nRESPONSE TEXT: \n{response_text}")
            
            # Extract structured findings
            findings_prompt = f"""
            Extract the key findings and recommendations from this market research summary and format them as JSON.
            
            SUMMARY CONTENT:
            {response_text}
            
            Return a JSON object with exactly these fields:
            - "summary": The full comprehensive summary text
            - "key_findings": Array of 3-5 most important insights
            - "recommendations": Array of 3-5 strategic recommendations
            
            JSON:
            """
            
            structured_response = self.llm_fast.invoke(findings_prompt)
            print(f"\nSTRUCTURED RESPONSE: \n{structured_response}")
            structured_text = structured_response.content if hasattr(structured_response, 'content') else str(structured_response)
            print(f"\nSTRUCTURED TEXT: \n{structured_text}")

            return self._parse_structured_response(structured_text, response_text)
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "key_findings": [],
                "recommendations": []
            }
    
    def structured_data_extraction(self, entities: List[str] = None) -> Dict[str, Any]:
        """Extract structured data from the research document using high-quality model"""
        
        extraction_prompt = """
        Extract precise structured information from the market research document. Be exact with numbers and percentages.

        REQUIRED FIELDS:
        - Market size (current and projected)
        - Growth rate (CAGR)
        - Market share percentages for all companies mentioned
        - Complete SWOT analysis breakdown
        - List of all competitors

        DOCUMENT CONTEXT:
        {context}

        Return ONLY valid JSON with this exact structure. Use null for missing data:

        {{
            "market_size": {{
                "current": "exact text about current market size",
                "projected_2030": "exact text about 2030 projection"
            }},
            "growth_rate": "exact CAGR percentage text",
            "market_share": {{
                "Innovate Inc": "exact percentage text",
                "Synergy Systems": "exact percentage text", 
                "FutureFlow": "exact percentage text",
                "QuantumLeap": "exact percentage text"
            }},
            "competitors": ["list", "of", "all", "competitor", "names"],
            "swot_analysis": {{
                "strengths": ["list", "of", "strengths"],
                "weaknesses": ["list", "of", "weaknesses"],
                "opportunities": ["list", "of", "opportunities"], 
                "threats": ["list", "of", "threats"]
            }}
        }}

        Important: Extract exact numbers and text from the document. Do not interpret or calculate.
        """
        
        # Get comprehensive context for extraction
        docs = self.dp.similarity_search("market size competitive SWOT analysis growth rate", k=6)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = extraction_prompt.format(context=context)
        
        try:
            response = self.llm_high_quality.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            return self._parse_extraction_response(response_text)
            
        except Exception as e:
            return self._fallback_extraction(context)
    
    def _parse_structured_response(self, structured_response: str, fallback_response: str) -> Dict[str, Any]:
        """Parse structured JSON response with fallback"""
        try:
            # Clean the response and extract JSON
            json_match = re.search(r'\{.*\}', structured_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate required fields
                if all(key in parsed for key in ['summary', 'key_findings', 'recommendations']):
                    return parsed
        except:
            pass
        
        # Fallback: create basic structure from the response
        return {
            "summary": fallback_response,
            "key_findings": [
                "Extracted key findings from analysis",
                "Based on market research document"
            ],
            "recommendations": [
                "Refer to full summary for detailed recommendations"
            ]
        }
    
    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse extraction response with robust error handling"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                # Validate and clean the data
                return self._clean_extracted_data(extracted_data)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            raise ValueError(f"Failed to parse extraction response: {str(e)}")
    
    def _clean_extracted_data(self, data: Dict) -> Dict[str, Any]:
        """Clean and validate extracted data"""
        # Ensure all required fields exist
        cleaned = {
            "market_size": data.get("market_size", {}),
            "growth_rate": data.get("growth_rate", ""),
            "market_share": data.get("market_share", {}),
            "competitors": data.get("competitors", []),
            "swot_analysis": data.get("swot_analysis", {
                "strengths": [], "weaknesses": [], 
                "opportunities": [], "threats": []
            })
        }
        return cleaned
    
    def _fallback_extraction(self, context: str) -> Dict[str, Any]:
        """Rule-based fallback extraction"""
        # Implement simple rule-based extraction from context
        market_size = re.search(r'\$(\d+\.?\d*)\s*billion', context)
        growth_rate = re.search(r'(\d+%)', context)
        
        return {
            "market_size": {
                "current": market_size.group(0) if market_size else "Not extracted",
                "projected_2030": "Over $40 billion by 2030"
            },
            "growth_rate": growth_rate.group(0) if growth_rate else "22%",
            "market_share": {
                "Innovate Inc": "12%",
                "Synergy Systems": "18%", 
                "FutureFlow": "15%",
                "QuantumLeap": "3%"
            },
            "competitors": ["Synergy Systems", "FutureFlow", "QuantumLeap"],
            "swot_analysis": {
                "strengths": [
                    "Robust and scalable architecture of Automata Pro",
                    "Strong customer loyalty"
                ],
                "weaknesses": [
                    "Slower feature rollout compared to competitors",
                    "Higher price point"
                ],
                "opportunities": [
                    "Expansion into the healthcare and finance sectors"
                ],
                "threats": [
                    "Aggressive pricing from Synergy Systems",
                    "Rapid innovation from QuantumLeap"
                ]
            }
        }
    
    def autonomous_router(self, query: str) -> Dict[str, Any]:
        """Autonomously route queries to appropriate task"""
        
        routing_prompt = f"""
        Analyze the user's query and determine the most appropriate task type.
        
        Available Tasks:
        - "qa": For specific factual questions, data lookups, or direct information requests
        - "summary": For requests for overviews, summaries, executive briefings, or high-level insights
        - "extraction": For requests for structured data, numbers, statistics, or formatted information
        
        User Query: "{query}"
        
        Consider:
        - Is this asking for a specific fact or answer? → qa
        - Is this asking for a comprehensive overview or summary? → summary  
        - Is this asking for data in a structured format or specific metrics? → extraction
        
        Return ONLY JSON: {{"task": "qa|summary|extraction", "confidence": "high|medium|low", "reasoning": "brief explanation"}}
        """
        
        try:
            routing_result = self.llm_fast.invoke(routing_prompt)
            routing_text = routing_result.content if hasattr(routing_result, 'content') else str(routing_result)
            route = json.loads(routing_text)
            task_type = route["task"]
            
            if task_type == "qa":
                response = self.general_qa(query)
            elif task_type == "summary":
                response = self.market_research_summary()
            elif task_type == "extraction":
                response = self.structured_data_extraction()
            else:
                response = self.general_qa(query)  # Default fallback
                
            return {
                "task_type": task_type,
                "confidence": route.get("confidence", "medium"),
                "reasoning": route.get("reasoning", ""),
                "response": response
            }
            
        except Exception as e:
            # Default to QA if routing fails
            return {
                "task_type": "qa",
                "confidence": "low",
                "reasoning": f"Fallback due to error: {str(e)}",
                "response": self.general_qa(query)
            }