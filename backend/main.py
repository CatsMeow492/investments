from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime, timedelta, timezone
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

# Model Configuration
MODEL_TIMEOUT = 60.0  # 60 seconds timeout for model requests

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

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    is_available: bool

class ResearchQuery(BaseModel):
    query: str
    context: ResearchContext
    should_use_web: bool = True
    model: Optional[str] = None  # Allow model selection

# AI Research Configuration
class AIModel:
    OPENAI = "openai"
    OPENROUTER_CLAUDE = "openrouter_claude"
    OPENROUTER_MISTRAL = "openrouter_mistral"
    OPENROUTER_DEEPSEEK = "openrouter_deepseek"
    ANTHROPIC = "anthropic"

    @staticmethod
    def get_available_models() -> List[ModelInfo]:
        """Get list of available models with their status"""
        models = [
            ModelInfo(
                id=AIModel.OPENAI,
                name="GPT-4 Turbo",
                description="OpenAI's most capable model, best for complex analysis",
                is_available=bool(OPENAI_API_KEY)
            ),
            ModelInfo(
                id=AIModel.OPENROUTER_CLAUDE,
                name="Claude 3 Opus",
                description="Anthropic's most capable model, excellent for detailed analysis and reasoning",
                is_available=bool(OPENROUTER_API_KEY)
            ),
            ModelInfo(
                id=AIModel.OPENROUTER_MISTRAL,
                name="Mixtral 8x7B",
                description="Fast and efficient model with strong reasoning capabilities",
                is_available=bool(OPENROUTER_API_KEY)
            ),
            ModelInfo(
                id=AIModel.OPENROUTER_DEEPSEEK,
                name="DeepSeek R1",
                description="Specialized model with strong coding and analysis capabilities",
                is_available=bool(OPENROUTER_API_KEY)
            ),
            ModelInfo(
                id=AIModel.ANTHROPIC,
                name="Claude 3 Opus (Direct)",
                description="Direct access to Claude 3 Opus via Anthropic's API",
                is_available=bool(ANTHROPIC_API_KEY)
            )
        ]
        return models

# Environment variables for different AI providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Default model configuration - prefer OpenRouter if available
DEFAULT_MODEL = AIModel.OPENROUTER_CLAUDE if OPENROUTER_API_KEY else (
    AIModel.OPENAI if OPENAI_API_KEY else AIModel.ANTHROPIC if ANTHROPIC_API_KEY else None
)

if not DEFAULT_MODEL:
    print("Warning: No API keys configured for any AI model")

# OpenRouter base configuration
OPENROUTER_BASE_CONFIG = {
    "headers": {
        "HTTP-Referer": "https://github.com/CatsMeow492/investments",
        "X-Title": "Investment Portfolio Tracker",
        "Content-Type": "application/json"
    },
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Model-specific configurations
MODEL_CONFIGS = {
    AIModel.OPENAI: {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4-turbo-preview",
        "max_tokens": 4000,
    },
    AIModel.OPENROUTER_CLAUDE: {
        **OPENROUTER_BASE_CONFIG,
        "model": "anthropic/claude-3-opus",
        "name": "Claude 3 Opus",
        "description": "Anthropic's most capable model, excellent for detailed analysis and reasoning",
    },
    AIModel.OPENROUTER_MISTRAL: {
        **OPENROUTER_BASE_CONFIG,
        "model": "mistralai/mixtral-8x7b",
        "name": "Mixtral 8x7B",
        "description": "Fast and efficient model with strong reasoning capabilities",
    },
    AIModel.OPENROUTER_DEEPSEEK: {
        **OPENROUTER_BASE_CONFIG,
        "model": "deepseek/deepseek-r1",
        "name": "DeepSeek R1",
        "description": "Specialized model with strong coding and analysis capabilities",
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

@app.get("/api/portfolio/update-prices")
async def update_portfolio_prices(db: Session = Depends(get_db)):
    """Update current prices for all assets in the portfolio"""
    try:
        assets = db.query(models.PortfolioAsset).all()
        updated_assets = set()  # Use a set to prevent duplicates
        failed_assets = []
        
        # Only update prices if they're older than 24 hours
        current_time = datetime.utcnow()
        
        # Group assets by those needing updates
        assets_needing_update = [
            asset for asset in assets
            if not asset.last_updated or 
            (current_time - asset.last_updated).total_seconds() >= 24 * 3600
        ]
        
        # Process in smaller batches to respect rate limits
        BATCH_SIZE = 5  # Process 5 assets at a time
        for i in range(0, len(assets_needing_update), BATCH_SIZE):
            batch = assets_needing_update[i:i + BATCH_SIZE]
            
            for asset in batch:
                try:
                    if asset.asset_type.lower() == 'stock':
                        url = f"https://api.polygon.io/v2/aggs/ticker/{asset.symbol}/prev?apiKey={os.getenv('POLYGON_API_KEY', 'sRu78wED36ZP1bwn8y20GfEj8URDCG5Y')}"
                    else:  # crypto
                        url = f"https://api.polygon.io/v2/aggs/ticker/X:{asset.symbol}USD/prev?apiKey={os.getenv('POLYGON_API_KEY', 'sRu78wED36ZP1bwn8y20GfEj8URDCG5Y')}"
                    
                    response = requests.get(url)
                    data = response.json()
                    
                    if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                        asset.current_price = float(data["results"][0]["c"])
                        asset.last_updated = current_time
                        db.add(asset)
                        updated_assets.add(asset)  # Add to set instead of list
                    else:
                        if "error" in data and "exceeded" in data["error"].lower():
                            # Rate limit hit, add delay and retry later
                            failed_assets.append({
                                "symbol": asset.symbol,
                                "reason": "Rate limit exceeded, will retry later"
                            })
                        else:
                            failed_assets.append({
                                "symbol": asset.symbol,
                                "reason": "No price data available"
                            })
                            
                except Exception as e:
                    print(f"Error updating price for {asset.symbol}: {str(e)}")
                    failed_assets.append({
                        "symbol": asset.symbol,
                        "reason": str(e)
                    })
                
                # Add delay between requests to respect rate limits
                await asyncio.sleep(0.5)  # 500ms delay between requests
            
            # Add a longer delay between batches
            await asyncio.sleep(2)  # 2 second delay between batches
            db.commit()  # Commit each batch
        
        # Add already up-to-date assets to updated_assets
        up_to_date_assets = [
            asset for asset in assets 
            if asset not in assets_needing_update
        ]
        updated_assets.update(up_to_date_assets)  # Use set.update instead of extend
        
        # Convert set to list for the response
        updated_assets_list = list(updated_assets)
        
        return {
            "message": f"Updated prices for {len(updated_assets_list)} assets. {len(failed_assets)} assets pending update.",
            "updated_assets": [{
                "symbol": asset.symbol,
                "current_price": asset.current_price,
                "last_updated": asset.last_updated
            } for asset in updated_assets_list],
            "failed_assets": failed_assets,
            "cache_info": {
                "next_update": (datetime.utcnow() + timedelta(minutes=1)).isoformat() if failed_assets else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prices: {str(e)}")

@app.get("/api/portfolio/assets")
async def get_portfolio_assets(db: Session = Depends(get_db)):
    """Get all assets in the portfolio with current values and gain/loss calculations"""
    try:
        assets = db.query(models.PortfolioAsset).all()
        
        # Check if prices need updating (older than 24 hours)
        assets_needing_update = [
            asset for asset in assets
            if not asset.last_updated or 
            (datetime.utcnow() - asset.last_updated).total_seconds() > 24 * 3600
        ]
        
        # Start update in background if needed, but don't wait for it
        if assets_needing_update:
            background_tasks = BackgroundTasks()
            background_tasks.add_task(update_portfolio_prices, db)
        
        return {
            "assets": [{
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
                "last_updated": asset.last_updated,
                "needs_update": not asset.last_updated or 
                    (datetime.utcnow() - asset.last_updated).total_seconds() > 24 * 3600
            } for asset in assets],
            "update_status": {
                "total_assets": len(assets),
                "assets_needing_update": len(assets_needing_update),
                "next_update_attempt": (datetime.utcnow() + timedelta(minutes=1)).isoformat() 
                    if assets_needing_update else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/allocation")
async def get_portfolio_allocation(db: Session = Depends(get_db)):
    """Get portfolio allocation by asset type and individual holdings"""
    portfolio = db.query(models.Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    assets = db.query(models.PortfolioAsset).filter(
        models.PortfolioAsset.portfolio_id == portfolio.id
    ).all()
    
    total_value = sum(asset.quantity * asset.current_price for asset in assets)
    
    # Group by asset type
    type_allocation = {}
    for asset in assets:
        value = asset.quantity * asset.current_price
        if asset.asset_type not in type_allocation:
            type_allocation[asset.asset_type] = 0
        type_allocation[asset.asset_type] += value
    
    # Calculate individual asset allocation
    holdings_allocation = []
    for asset in assets:
        value = asset.quantity * asset.current_price
        percentage = (value / total_value * 100) if total_value > 0 else 0
        if percentage >= 0.1:  # Only include holdings that are 0.1% or more of the portfolio
            holdings_allocation.append({
                "symbol": asset.symbol,
                "name": asset.name,
                "value": value,
                "percentage": percentage
            })
    
    # Sort holdings by percentage in descending order
    holdings_allocation.sort(key=lambda x: x["percentage"], reverse=True)
    
    # Convert type allocation to percentages
    type_allocation_percentages = {
        asset_type: (value / total_value * 100) if total_value > 0 else 0
        for asset_type, value in type_allocation.items()
    }
    
    return {
        "total_value": total_value,
        "by_type": type_allocation_percentages,
        "by_holding": holdings_allocation
    }

@app.get("/api/stocks/{symbol}")
async def get_stock_price(symbol: str):
    """Get current stock price using Polygon.io API"""
    api_key = os.getenv("POLYGON_API_KEY", "sRu78wED36ZP1bwn8y20GfEj8URDCG5Y")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data["status"] != "OK" or data["resultsCount"] == 0:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        price = float(data["results"][0]["c"])
        return {
            "symbol": symbol,
            "price": price,
            "last_updated": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/{symbol}")
async def get_crypto_price(symbol: str):
    """Get current cryptocurrency price using Polygon.io API"""
    api_key = os.getenv("POLYGON_API_KEY", "sRu78wED36ZP1bwn8y20GfEj8URDCG5Y")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/prev?apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data["status"] != "OK" or data["resultsCount"] == 0:
            raise HTTPException(status_code=404, detail="Cryptocurrency not found")
        
        price = float(data["results"][0]["c"])
        return AssetPrice(
            symbol=symbol,
            price=price,
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_web_data(symbol: str) -> Dict[str, Any]:
    """Fetch latest news and market data for a given symbol"""
    print(f"Starting data fetch for {symbol}...")
    
    # Initialize results dictionary
    results = {
        "market_data": None,
        "news": None,
        "analyst_ratings": None
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # 1. First fetch market data as it's most critical
            polygon_key = os.getenv('POLYGON_API_KEY', 'sRu78wED36ZP1bwn8y20GfEj8URDCG5Y')
            market_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={polygon_key}"
            
            print(f"Fetching market data for {symbol}...")
            market_response = await client.get(market_url)
            
            if market_response.status_code == 200:
                market_json = market_response.json()
                if market_json.get("status") == "OK" and market_json.get("resultsCount", 0) > 0:
                    results["market_data"] = {
                        "price": float(market_json["results"][0]["c"]),
                        "open": float(market_json["results"][0]["o"]),
                        "high": float(market_json["results"][0]["h"]),
                        "low": float(market_json["results"][0]["l"]),
                        "volume": float(market_json["results"][0]["v"])
                    }
                else:
                    print(f"No market data available for {symbol}: {market_json.get('error')}")
            
            # Add delay between API calls
            await asyncio.sleep(1)
            
            # 2. Then fetch news if we have the API key
            alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if alpha_vantage_key:
                try:
                    print(f"Fetching news for {symbol}...")
                    news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={alpha_vantage_key}"
                    news_response = await client.get(news_url)
                    
                    if news_response.status_code == 200:
                        news_data = news_response.json()
                        if "feed" in news_data:
                            results["news"] = []
                            for article in news_data["feed"][:5]:
                                results["news"].append({
                                    "title": article.get("title"),
                                    "summary": article.get("summary"),
                                    "url": article.get("url"),
                                    "sentiment": article.get("overall_sentiment_label"),
                                    "published_at": article.get("time_published")
                                })
                        else:
                            print(f"No news data available for {symbol}")
                except Exception as e:
                    print(f"Error fetching news for {symbol}: {str(e)}")
            
            # Add delay between API calls
            await asyncio.sleep(1)
            
            # 3. Finally fetch analyst ratings
            try:
                print(f"Fetching analyst ratings for {symbol}...")
                snapshot_url = f"https://api.polygon.io/v3/snapshot/ticker/{symbol}?apiKey={polygon_key}"
                snapshot_response = await client.get(snapshot_url)
                
                if snapshot_response.status_code == 200:
                    snapshot_json = snapshot_response.json()
                    if snapshot_json.get("results"):  # Changed from data to results based on Polygon.io API
                        data = snapshot_json["results"]
                        results["analyst_ratings"] = {
                            "price_target": data.get("price_target", {}).get("average"),
                            "recommendations": data.get("recommendations", {}).get("summary")
                        }
            except Exception as e:
                print(f"Error fetching analyst ratings for {symbol}: {str(e)}")
            
            print(f"Completed data fetch for {symbol}")
            return results
            
        except Exception as e:
            print(f"Error in fetch_web_data for {symbol}: {str(e)}")
            return results

async def query_ai_model(prompt: str, context: Dict[str, Any], model: str = None) -> Dict[str, Any]:
    """Query AI model with prompt and context"""
    # Use specified model or default to OpenRouter Claude if available
    if not model:
        model = DEFAULT_MODEL
        if not model:
            raise HTTPException(
                status_code=500,
                detail="No AI model API keys configured. Please configure OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
            )

    if model not in MODEL_CONFIGS:
        print(f"Error: Unsupported model {model}")
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    config = MODEL_CONFIGS[model]
    
    # Handle API key validation based on model type
    api_key = None
    if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
        api_key = OPENROUTER_API_KEY
    else:
        api_key = os.getenv(f"{model.upper()}_API_KEY")
    
    if not api_key:
        print(f"Error: Missing API key for {model}")
        raise HTTPException(status_code=500, detail=f"{model} API key not configured")

    print(f"Using model: {model} with config: {config['model']}")
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Model-specific header configuration
    if model == AIModel.OPENAI:
        headers["Authorization"] = f"Bearer {api_key}"
    elif model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
        headers["Authorization"] = f"Bearer {api_key}"
        headers.update(config.get("headers", {}))
    elif model == AIModel.ANTHROPIC:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2024-01-01"

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
            if model in [AIModel.OPENAI, AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
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
            print(f"Request headers: {headers}")
            print(f"Request data: {json.dumps(request_data, indent=2)}")
            
            response = await client.post(
                config["api_url"],
                headers=headers,
                json=request_data
            )
            
            print(f"{model} API response status: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content based on model
                if model in [AIModel.OPENAI, AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
                    content = result["choices"][0]["message"]["content"]
                elif model == AIModel.ANTHROPIC:
                    # Handle Anthropic's response format
                    content = result["messages"][0]["content"][0]["text"]
                
                return {
                    "answer": content,
                    "sources": formatted_context,
                    "model_used": config["model"]
                }
            else:
                error_message = response.text
                print(f"{model} API error: {error_message}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"{model} API error: {error_message}"
                )
                
    except httpx.TimeoutException as e:
        print(f"Timeout error: {str(e)}")
        raise HTTPException(status_code=504, detail=f"Request to {model} API timed out")
    except httpx.RequestError as e:
        print(f"Network error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Network error when contacting {model} API: {str(e)}")
    except Exception as e:
        print(f"Unexpected error querying {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying {model}: {str(e)}")

@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models"""
    return AIModel.get_available_models()

async def stream_ai_response(prompt: str, context: Dict[str, Any], model: str = None) -> AsyncGenerator[str, None]:
    """Stream AI model response"""
    try:
        # Use specified model or default
        if not model:
            model = DEFAULT_MODEL
            if not model:
                raise HTTPException(
                    status_code=500,
                    detail="No AI model API keys configured"
                )

        config = MODEL_CONFIGS[model]
        print(f"Streaming with model: {model}, config: {config['model']}")
        
        # Handle API key validation
        api_key = None
        if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
            api_key = OPENROUTER_API_KEY
        else:
            api_key = os.getenv(f"{model.upper()}_API_KEY")
        
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail=f"{model} API key not configured"
            )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
            headers.update(config.get("headers", {}))
        elif model == AIModel.ANTHROPIC:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2024-01-01"

        # Prepare the system message and request data
        system_message = """You are an AI investment research assistant. Analyze the provided portfolio and market data to give informed insights and recommendations. Base your analysis on:
1. Portfolio composition and performance
2. Market conditions and trends
3. Risk factors and opportunities
4. Latest news and analyst opinions
Always provide specific, data-backed insights and clear reasoning for your recommendations."""

        request_data = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context: {json.dumps(context, indent=2)}\n\nQuery: {prompt}"}
            ],
            "stream": True,  # Enable streaming
            "temperature": 0.7,
            "max_tokens": config["max_tokens"],
        }

        print(f"Sending streaming request to {config['api_url']}")
        async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
            async with client.stream(
                "POST",
                config["api_url"],
                headers=headers,
                json=request_data,
                timeout=MODEL_TIMEOUT
            ) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.aiter_bytes():
                    chunk_str = chunk.decode()
                    print(f"Received chunk: {chunk_str[:100]}...")  # Debug log
                    
                    # Handle SSE format
                    if chunk_str.startswith("data: "):
                        chunk_str = chunk_str[6:]
                    
                    try:
                        # Try to parse as JSON
                        data = json.loads(chunk_str)
                        
                        # Extract content based on model type
                        content = None
                        if model in [AIModel.OPENROUTER_CLAUDE, AIModel.OPENROUTER_MISTRAL, AIModel.OPENROUTER_DEEPSEEK]:
                            if "choices" in data and data["choices"]:
                                content = data["choices"][0].get("delta", {}).get("content", "")
                        elif model == AIModel.OPENAI:
                            if "choices" in data and data["choices"]:
                                content = data["choices"][0].get("delta", {}).get("content", "")
                        elif model == AIModel.ANTHROPIC:
                            if "type" in data and data["type"] == "content_block_delta":
                                content = data.get("delta", {}).get("text", "")
                        
                        if content:
                            yield json.dumps({
                                "answer": content,
                                "model_used": config["model"]
                            }) + "\n"
                            
                    except json.JSONDecodeError:
                        # If we can't parse as JSON, accumulate in buffer
                        buffer += chunk_str
                        if "\n" in buffer:
                            lines = buffer.split("\n")
                            buffer = lines[-1]  # Keep the incomplete line
                            
                            for line in lines[:-1]:
                                if line.strip():
                                    try:
                                        if line.startswith("data: "):
                                            line = line[6:]
                                        data = json.loads(line)
                                        content = None
                                        if "choices" in data and data["choices"]:
                                            content = data["choices"][0].get("delta", {}).get("content", "")
                                        if content:
                                            yield json.dumps({
                                                "answer": content,
                                                "model_used": config["model"]
                                            }) + "\n"
                                    except json.JSONDecodeError:
                                        continue

    except Exception as e:
        print(f"Streaming error: {str(e)}")
        yield json.dumps({"error": str(e)}) + "\n"

@app.post("/api/research/query")
async def research_query(
    query: ResearchQuery,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Process a research query using AI models"""
    try:
        # Verify at least one AI model is configured
        if not any([OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY]):
            raise HTTPException(
                status_code=500,
                detail="No AI model API keys configured. Please configure at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
            )

        # Initialize context variables
        portfolio_data = None
        asset_data = None
        web_data = None
        assets = []

        try:
            # If context includes portfolio, fetch portfolio data
            if query.context.include_portfolio:
                portfolio = db.query(models.Portfolio).first()
                if not portfolio:
                    print("Warning: No portfolio found in database")
                else:
                    assets = db.query(models.PortfolioAsset).filter(
                        models.PortfolioAsset.portfolio_id == portfolio.id
                    ).all()
                    
                    if not assets:
                        print("Warning: No assets found in portfolio")
                    else:
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
                                "current_value": asset.quantity * asset.current_price,
                                "weight": (asset.quantity * asset.current_price / total_value * 100) if total_value > 0 else 0,
                            } for asset in assets]
                        }

        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            portfolio_data = None
            assets = []

        # If context includes specific asset, fetch asset data
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
                    "current_value": asset.quantity * asset.current_price,
                    "portfolio_weight": (asset.quantity * asset.current_price / total_value * 100) if total_value > 0 else 0,
                }

        # Fetch web data if requested
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

        # Prepare context for AI model
        research_context = {
            "query": query.query,
            "portfolio": portfolio_data,
            "asset": asset_data,
            "web_data": web_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Return streaming response
        return StreamingResponse(
            stream_ai_response(query.query, research_context, query.model),
            media_type="text/event-stream"
        )

    except Exception as e:
        print(f"Error in research query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/recommendations")
async def get_portfolio_recommendations(db: Session = Depends(get_db)):
    """Get AI-powered recommendations for portfolio holdings"""
    try:
        assets = db.query(models.PortfolioAsset).all()
        recommendations = []
        
        # Process assets in smaller batches to avoid overwhelming APIs
        BATCH_SIZE = 3
        for i in range(0, len(assets), BATCH_SIZE):
            batch = assets[i:i + BATCH_SIZE]
            
            for asset in batch:
                try:
                    print(f"Processing recommendations for {asset.symbol}...")
                    # Fetch latest news and market data
                    web_data = await fetch_web_data(asset.symbol)
                    
                    # Prepare context for AI analysis
                    asset_context = {
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "type": asset.asset_type,
                        "current_price": asset.current_price,
                        "market_data": web_data.get("market_data"),
                        "news": web_data.get("news"),
                        "analyst_ratings": web_data.get("analyst_ratings")
                    }
                    
                    recommendation = {
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "value": asset.quantity * asset.current_price,
                        "market_data": web_data.get("market_data"),
                        "news": web_data.get("news", [])[:3] if web_data.get("news") else [],
                        "analyst_ratings": web_data.get("analyst_ratings"),
                        "last_updated": datetime.now(timezone.utc).isoformat()  # Use timezone-aware datetime
                    }
                    
                    # Only proceed with AI analysis if we have some data to work with
                    if web_data.get("market_data") or web_data.get("news") or web_data.get("analyst_ratings"):
                        print(f"Querying AI for {asset.symbol}...")
                        
                        # Construct prompt based on available data
                        prompt_parts = [f"Analyze {asset.symbol} ({asset.name}) based on the following data:"]
                        
                        if web_data.get("market_data"):
                            market_data = web_data["market_data"]
                            prompt_parts.append(f"""
                            Market Data:
                            - Current Price: ${market_data['price']}
                            - Today's Range: ${market_data['low']} - ${market_data['high']}
                            - Volume: {market_data['volume']}""")
                        
                        if web_data.get("news"):
                            prompt_parts.append(f"News Headlines: {len(web_data['news'])} recent articles")
                        
                        if web_data.get("analyst_ratings", {}).get("price_target"):
                            prompt_parts.append(f"Analyst Ratings: Price Target ${web_data['analyst_ratings']['price_target']}")
                        
                        prompt_parts.append("""
                        Provide a concise recommendation with:
                        1. Overall sentiment (bullish/neutral/bearish)
                        2. Key factors influencing the recommendation
                        3. Suggested action (hold/watch/research)
                        4. Risk level (low/medium/high)""")
                        
                        prompt = "\n".join(prompt_parts)
                        ai_response = await query_ai_model(prompt, asset_context)
                        recommendation["analysis"] = ai_response["answer"]
                    else:
                        recommendation["analysis"] = "Insufficient data available for AI analysis. Please try again later."
                        recommendation["error"] = "No market data or news available"
                    
                    recommendations.append(recommendation)
                    
                except Exception as e:
                    print(f"Error analyzing {asset.symbol}: {str(e)}")
                    recommendations.append({
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "value": asset.quantity * asset.current_price,
                        "analysis": "Error generating recommendations. Please try again later.",
                        "error": str(e),
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    })
                
                # Add delay between processing assets
                await asyncio.sleep(2)
            
            # Add longer delay between batches
            if i + BATCH_SIZE < len(assets):
                print("Waiting between batches to respect API rate limits...")
                await asyncio.sleep(5)
        
        return {
            "recommendations": recommendations,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "error_count": len([r for r in recommendations if "error" in r])
        }
        
    except Exception as e:
        print(f"Error in get_portfolio_recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Error generating portfolio recommendations",
                "status": "failed"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 