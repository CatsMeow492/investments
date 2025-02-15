from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from schemas import Portfolio, Asset, PortfolioPerformance
import httpx
import json
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# DeepSeek R1 Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# DeepSeek model configuration
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT = 60.0

# Web scraping configuration
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

app = FastAPI(title="Investment Portfolio Tracker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class Asset(BaseModel):
    symbol: str
    name: str
    type: str  # 'stock' or 'crypto'
    quantity: float
    purchase_price: float
    purchase_date: datetime

class AssetPrice(BaseModel):
    symbol: str
    price: float
    last_updated: datetime

class ResearchContext(BaseModel):
    include_portfolio: bool = False
    symbol: Optional[str] = None

class ResearchQuery(BaseModel):
    query: str
    context: ResearchContext
    should_use_web: bool = True

# AI Research Configuration
class AIModel:
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

# Environment variables for different AI providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Default model configuration
DEFAULT_MODEL = AIModel.OPENAI if OPENAI_API_KEY else (
    AIModel.GEMINI if GEMINI_API_KEY else AIModel.ANTHROPIC
)
MODEL_TIMEOUT = 60.0

# Model-specific configurations
MODEL_CONFIGS = {
    AIModel.OPENAI: {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4-turbo-preview",
        "max_tokens": 4000,
    },
    AIModel.GEMINI: {
        "api_url": "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
        "max_tokens": 2048,
    },
    AIModel.ANTHROPIC: {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-opus-20240229",
        "max_tokens": 4000,
    }
}

# Routes
@app.get("/")
async def read_root():
    return {"message": "Welcome to Investment Portfolio Tracker API"}

@app.get("/api/portfolio/summary")
async def get_portfolio_summary(db: Session = Depends(get_db)):
    """Get portfolio summary including total value, gain/loss, and return"""
    portfolio = db.query(models.Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    assets = db.query(models.PortfolioAsset).filter(
        models.PortfolioAsset.portfolio_id == portfolio.id
    ).all()
    
    total_value = sum(asset.quantity * asset.current_price for asset in assets)
    total_cost = sum(asset.quantity * asset.purchase_price for asset in assets)
    total_gain_loss = total_value - total_cost
    gain_loss_percentage = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
    
    return {
        "total_value": total_value,
        "total_gain_loss": total_gain_loss,
        "gain_loss_percentage": gain_loss_percentage,
        "last_updated": datetime.utcnow()
    }

@app.get("/api/portfolio/assets")
async def get_portfolio_assets(db: Session = Depends(get_db)):
    """Get all assets in the portfolio"""
    portfolio = db.query(models.Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    assets = db.query(models.PortfolioAsset).filter(
        models.PortfolioAsset.portfolio_id == portfolio.id
    ).all()
    
    return [{
        "symbol": asset.symbol,
        "name": asset.name,
        "type": asset.asset_type,
        "quantity": asset.quantity,
        "current_price": asset.current_price,
        "purchase_price": asset.purchase_price,
        "current_value": asset.quantity * asset.current_price,
        "gain_loss": (asset.current_price - asset.purchase_price) * asset.quantity,
        "gain_loss_percentage": ((asset.current_price - asset.purchase_price) / asset.purchase_price * 100) 
            if asset.purchase_price > 0 else 0,
        "last_updated": asset.last_updated
    } for asset in assets]

@app.get("/api/portfolio/allocation")
async def get_portfolio_allocation(db: Session = Depends(get_db)):
    """Get portfolio allocation by asset type"""
    portfolio = db.query(models.Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    assets = db.query(models.PortfolioAsset).filter(
        models.PortfolioAsset.portfolio_id == portfolio.id
    ).all()
    
    total_value = sum(asset.quantity * asset.current_price for asset in assets)
    
    # Group by asset type
    allocation = {}
    for asset in assets:
        value = asset.quantity * asset.current_price
        if asset.asset_type not in allocation:
            allocation[asset.asset_type] = 0
        allocation[asset.asset_type] += value
    
    # Convert to percentages
    for asset_type in allocation:
        allocation[asset_type] = (allocation[asset_type] / total_value * 100) if total_value > 0 else 0
    
    return allocation

@app.get("/api/stocks/{symbol}")
async def get_stock_price(symbol: str):
    """Get current stock price using Alpha Vantage API"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Alpha Vantage API key not configured")
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Global Quote" not in data:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        price = float(data["Global Quote"]["05. price"])
        return {
            "symbol": symbol,
            "price": price,
            "last_updated": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/{symbol}")
async def get_crypto_price(symbol: str):
    """Get current cryptocurrency price using CoinGecko API"""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if symbol not in data:
            raise HTTPException(status_code=404, detail="Cryptocurrency not found")
        
        price = float(data[symbol]["usd"])
        return AssetPrice(
            symbol=symbol,
            price=price,
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_web_data(symbol: str) -> Dict[str, Any]:
    """Fetch latest news and market data for a given symbol"""
    async with httpx.AsyncClient() as client:
        # Fetch from multiple sources concurrently
        tasks = [
            fetch_yahoo_finance(client, symbol),
            fetch_market_news(client, symbol),
            fetch_analyst_ratings(client, symbol)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        web_data = {
            "market_data": results[0] if not isinstance(results[0], Exception) else None,
            "news": results[1] if not isinstance(results[1], Exception) else None,
            "analyst_ratings": results[2] if not isinstance(results[2], Exception) else None
        }
        
        return web_data

async def fetch_yahoo_finance(client: httpx.AsyncClient, symbol: str) -> Dict[str, Any]:
    """Fetch market data from Yahoo Finance"""
    url = f"https://finance.yahoo.com/quote/{quote_plus(symbol)}"
    try:
        response = await client.get(url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Extract relevant data (customize based on needs)
            data = {
                "price": extract_text(soup, '[data-test="qsp-price"]'),
                "change": extract_text(soup, '[data-test="qsp-price-change"]'),
                "market_cap": extract_text(soup, '[data-test="market-cap"]'),
                "pe_ratio": extract_text(soup, '[data-test="PE_RATIO-value"]'),
                "volume": extract_text(soup, '[data-test="VOLUME-value"]')
            }
            return data
    except Exception as e:
        print(f"Error fetching Yahoo Finance data: {e}")
        return None

async def fetch_market_news(client: httpx.AsyncClient, symbol: str) -> List[Dict[str, str]]:
    """Fetch latest news articles"""
    url = f"https://api.marketaux.com/v1/news/all?symbols={symbol}&api_token={os.getenv('MARKETAUX_API_KEY')}"
    try:
        response = await client.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])[:5]  # Return top 5 news articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return None

async def fetch_analyst_ratings(client: httpx.AsyncClient, symbol: str) -> Dict[str, Any]:
    """Fetch analyst ratings and price targets"""
    url = f"https://www.tipranks.com/stocks/{symbol.lower()}/forecast"
    try:
        response = await client.get(url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Extract analyst consensus and price targets
            data = {
                "consensus": extract_analyst_consensus(soup),
                "price_target": extract_price_target(soup)
            }
            return data
    except Exception as e:
        print(f"Error fetching analyst ratings: {e}")
        return None

def extract_text(soup: BeautifulSoup, selector: str) -> str:
    """Helper function to extract text from BeautifulSoup object"""
    element = soup.select_one(selector)
    return element.text.strip() if element else None

def extract_analyst_consensus(soup: BeautifulSoup) -> str:
    """Extract analyst consensus from TipRanks"""
    consensus_element = soup.find("div", {"data-test": "analyst-consensus"})
    return consensus_element.text.strip() if consensus_element else None

def extract_price_target(soup: BeautifulSoup) -> str:
    """Extract price target from TipRanks"""
    target_element = soup.find("div", {"data-test": "price-target"})
    return target_element.text.strip() if target_element else None

async def query_ai_model(prompt: str, context: Dict[str, Any], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Query AI model with prompt and context"""
    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    config = MODEL_CONFIGS[model]
    api_key = os.getenv(f"{model.upper()}_API_KEY")
    
    if not api_key:
        raise HTTPException(status_code=500, detail=f"{model} API key not configured")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Model-specific header configuration
    if model == AIModel.OPENAI:
        headers["Authorization"] = f"Bearer {api_key}"
    elif model == AIModel.ANTHROPIC:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2024-01-01"
    elif model == AIModel.GEMINI:
        headers["Authorization"] = f"Bearer {api_key}"

    # Prepare the system message
    system_message = """You are an AI investment research assistant. Analyze the provided portfolio and market data to give informed insights and recommendations. Base your analysis on:
1. Portfolio composition and performance
2. Market conditions and trends
3. Risk factors and opportunities
4. Latest news and analyst opinions
Always provide specific, data-backed insights and clear reasoning for your recommendations."""

    # Format context
    formatted_context = {
        "query_type": "portfolio" if context.get("portfolio") else "asset",
        "portfolio_summary": context.get("portfolio", {}).get("summary") if context.get("portfolio") else None,
        "asset_details": context.get("asset") if context.get("asset") else None,
        "market_data": context.get("web_data", {}).get("market_data") if context.get("web_data") else None,
        "latest_news": context.get("web_data", {}).get("news") if context.get("web_data") else None,
        "analyst_ratings": context.get("web_data", {}).get("analyst_ratings") if context.get("web_data") else None
    }

    try:
        async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
            # Prepare request based on model
            if model == AIModel.OPENAI:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context: {json.dumps(formatted_context, indent=2)}\n\nQuery: {prompt}"}
                ]
                request_data = {
                    "model": config["model"],
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": config["max_tokens"],
                }
            elif model == AIModel.GEMINI:
                request_data = {
                    "contents": [{
                        "role": "user",
                        "parts": [{
                            "text": f"{system_message}\n\nContext: {json.dumps(formatted_context, indent=2)}\n\nQuery: {prompt}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": config["max_tokens"],
                    }
                }
            elif model == AIModel.ANTHROPIC:
                request_data = {
                    "model": config["model"],
                    "messages": [{
                        "role": "user",
                        "content": f"{system_message}\n\nContext: {json.dumps(formatted_context, indent=2)}\n\nQuery: {prompt}"
                    }],
                    "max_tokens": config["max_tokens"]
                }

            print(f"Sending request to {model} API: {config['api_url']}")
            response = await client.post(
                config["api_url"],
                headers=headers,
                json=request_data
            )
            
            print(f"{model} API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content based on model
                if model == AIModel.OPENAI:
                    content = result["choices"][0]["message"]["content"]
                elif model == AIModel.GEMINI:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                elif model == AIModel.ANTHROPIC:
                    content = result["content"][0]["text"]
                
                return {
                    "answer": content,
                    "sources": formatted_context,
                    "model_used": model
                }
            else:
                error_message = response.text
                print(f"{model} API error: {error_message}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"{model} API error: {error_message}"
                )
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Request to {model} API timed out")
    except httpx.RequestError as e:
        print(f"Network error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Network error when contacting {model} API: {str(e)}")
    except Exception as e:
        print(f"Unexpected error querying {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying {model}: {str(e)}")

@app.post("/api/research/query")
async def research_query(
    query: ResearchQuery,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    model: str = DEFAULT_MODEL
):
    """Process a research query using AI models"""
    try:
        # If context includes portfolio, fetch portfolio data
        portfolio_data = None
        portfolio_summary = None
        if query.context.include_portfolio:
            portfolio = db.query(models.Portfolio).first()
            if portfolio:
                assets = db.query(models.PortfolioAsset).filter(
                    models.PortfolioAsset.portfolio_id == portfolio.id
                ).all()
                
                # Calculate portfolio summary
                total_value = sum(asset.quantity * asset.current_price for asset in assets)
                total_cost = sum(asset.quantity * asset.purchase_price for asset in assets)
                total_gain_loss = total_value - total_cost
                gain_loss_percentage = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
                
                # Group assets by type
                asset_types = {}
                for asset in assets:
                    if asset.asset_type not in asset_types:
                        asset_types[asset.asset_type] = 0
                    asset_types[asset.asset_type] += asset.quantity * asset.current_price
                
                # Calculate type allocations
                type_allocation = {
                    asset_type: (value / total_value * 100) if total_value > 0 else 0
                    for asset_type, value in asset_types.items()
                }
                
                # Sort assets by value
                asset_values = [(asset, asset.quantity * asset.current_price) for asset in assets]
                asset_values.sort(key=lambda x: x[1], reverse=True)
                
                portfolio_data = {
                    "summary": {
                        "total_value": total_value,
                        "total_cost": total_cost,
                        "total_gain_loss": total_gain_loss,
                        "gain_loss_percentage": gain_loss_percentage,
                        "asset_count": len(assets),
                        "type_allocation": type_allocation
                    },
                    "assets": [{
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "type": asset.asset_type,
                        "quantity": asset.quantity,
                        "current_price": asset.current_price,
                        "purchase_price": asset.purchase_price,
                        "current_value": asset.quantity * asset.current_price,
                        "weight": (asset.quantity * asset.current_price / total_value * 100) if total_value > 0 else 0,
                        "gain_loss": (asset.current_price - asset.purchase_price) * asset.quantity,
                        "gain_loss_percentage": ((asset.current_price - asset.purchase_price) / asset.purchase_price * 100) 
                            if asset.purchase_price > 0 else 0,
                    } for asset, _ in asset_values]
                }

        # If context includes specific asset, fetch asset data
        asset_data = None
        if query.context.symbol:
            asset = db.query(models.PortfolioAsset).filter(
                models.PortfolioAsset.symbol == query.context.symbol
            ).first()
            if asset:
                total_value = sum(a.quantity * a.current_price for a in assets)
                asset_data = {
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "type": asset.asset_type,
                    "quantity": asset.quantity,
                    "current_price": asset.current_price,
                    "purchase_price": asset.purchase_price,
                    "current_value": asset.quantity * asset.current_price,
                    "portfolio_weight": (asset.quantity * asset.current_price / total_value * 100) if total_value > 0 else 0,
                    "gain_loss": (asset.current_price - asset.purchase_price) * asset.quantity,
                    "gain_loss_percentage": ((asset.current_price - asset.purchase_price) / asset.purchase_price * 100) 
                        if asset.purchase_price > 0 else 0,
                }

        # Fetch web data if requested
        web_data = None
        if query.should_use_web:
            if query.context.symbol:
                web_data = await fetch_web_data(query.context.symbol)
            elif query.context.include_portfolio and portfolio_data:
                # Fetch data for top holdings
                top_holdings = portfolio_data["assets"][:3]  # Top 3 holdings
                web_data = {
                    asset["symbol"]: await fetch_web_data(asset["symbol"])
                    for asset in top_holdings
                }

        # Prepare context for DeepSeek
        research_context = {
            "query": query.query,
            "portfolio": portfolio_data,
            "asset": asset_data,
            "web_data": web_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Query AI model
        response = await query_ai_model(query.query, research_context, model)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 