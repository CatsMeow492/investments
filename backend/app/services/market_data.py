import httpx
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import os
from ..core.config import settings
from ..core.cache import get_cached_data, set_cached_data

class PolygonRateLimiter:
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        async with self.lock:
            current_time = datetime.now(timezone.utc)
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls 
                         if (current_time - call_time).total_seconds() < 60]
            
            if len(self.calls) >= self.calls_per_minute:
                # Wait until the oldest call is more than 1 minute old
                wait_time = 60 - (current_time - self.calls[0]).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                self.calls = self.calls[1:]  # Remove the oldest call
            
            self.calls.append(current_time)

# Create a global rate limiter instance
polygon_rate_limiter = PolygonRateLimiter(calls_per_minute=settings.POLYGON_CALLS_PER_MINUTE)

async def fetch_web_data(symbol: str) -> Dict[str, Any]:
    """Fetch latest news and market data for a given symbol with caching"""
    print(f"Starting data fetch for {symbol}...")
    
    # Initialize results dictionary
    results = {
        "market_data": None,
        "news": None,
        "analyst_ratings": None
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # 1. Try to get market data from cache first
            cache_key = f"market_data:{symbol}"
            cached_market_data = await get_cached_data(cache_key, "MARKET_DATA")
            
            if cached_market_data:
                results["market_data"] = cached_market_data
                print(f"Using cached market data for {symbol}")
            else:
                # Fetch fresh market data
                await polygon_rate_limiter.wait_if_needed()
                
                market_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={settings.POLYGON_API_KEY}"
                print(f"Fetching market data for {symbol}...")
                market_response = await client.get(market_url)
                
                if market_response.status_code == 200:
                    market_json = market_response.json()
                    if market_json.get("status") == "OK" and market_json.get("resultsCount", 0) > 0:
                        market_data = {
                            "price": float(market_json["results"][0]["c"]),
                            "open": float(market_json["results"][0]["o"]),
                            "high": float(market_json["results"][0]["h"]),
                            "low": float(market_json["results"][0]["l"]),
                            "volume": float(market_json["results"][0]["v"])
                        }
                        results["market_data"] = market_data
                        await set_cached_data(cache_key, market_data, "MARKET_DATA")
                    else:
                        print(f"No market data available for {symbol}: {market_json.get('error')}")
                elif market_response.status_code == 429:
                    print(f"Rate limit exceeded for {symbol}, using cached data if available...")
                    if cached_market_data:
                        results["market_data"] = cached_market_data
            
            # 2. Try to get news from cache
            news_cache_key = f"news:{symbol}"
            cached_news = await get_cached_data(news_cache_key, "NEWS")
            
            if cached_news:
                results["news"] = cached_news
                print(f"Using cached news for {symbol}")
            else:
                # Fetch fresh news if Alpha Vantage API key is available
                if settings.ALPHA_VANTAGE_API_KEY:
                    try:
                        print(f"Fetching news for {symbol}...")
                        news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={settings.ALPHA_VANTAGE_API_KEY}"
                        news_response = await client.get(news_url)
                        
                        if news_response.status_code == 200:
                            news_data = news_response.json()
                            if "feed" in news_data:
                                news = []
                                for article in news_data["feed"][:5]:
                                    news.append({
                                        "title": article.get("title"),
                                        "summary": article.get("summary"),
                                        "url": article.get("url"),
                                        "sentiment": article.get("overall_sentiment_label"),
                                        "published_at": article.get("time_published")
                                    })
                                results["news"] = news
                                await set_cached_data(news_cache_key, news, "NEWS")
                            else:
                                print(f"No news data available for {symbol}")
                    except Exception as e:
                        print(f"Error fetching news for {symbol}: {str(e)}")
            
            # 3. Try to get analyst ratings from cache
            if results["market_data"]:
                ratings_cache_key = f"analyst_ratings:{symbol}"
                cached_ratings = await get_cached_data(ratings_cache_key, "ANALYST_RATINGS")
                
                if cached_ratings:
                    results["analyst_ratings"] = cached_ratings
                    print(f"Using cached analyst ratings for {symbol}")
                else:
                    try:
                        print(f"Fetching analyst ratings for {symbol}...")
                        await polygon_rate_limiter.wait_if_needed()
                        
                        snapshot_url = f"https://api.polygon.io/v3/snapshot/ticker/{symbol}?apiKey={settings.POLYGON_API_KEY}"
                        snapshot_response = await client.get(snapshot_url)
                        
                        if snapshot_response.status_code == 200:
                            snapshot_json = snapshot_response.json()
                            if snapshot_json.get("results"):
                                data = snapshot_json["results"]
                                ratings = {
                                    "price_target": data.get("price_target", {}).get("average"),
                                    "recommendations": data.get("recommendations", {}).get("summary")
                                }
                                results["analyst_ratings"] = ratings
                                await set_cached_data(ratings_cache_key, ratings, "ANALYST_RATINGS")
                        elif snapshot_response.status_code == 429:
                            print(f"Rate limit exceeded for analyst ratings, using cached data if available...")
                            if cached_ratings:
                                results["analyst_ratings"] = cached_ratings
                    except Exception as e:
                        print(f"Error fetching analyst ratings for {symbol}: {str(e)}")
            
            print(f"Completed data fetch for {symbol}")
            return results
            
        except Exception as e:
            print(f"Error in fetch_web_data for {symbol}: {str(e)}")
            return results

async def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get current stock price using Polygon.io API"""
    await polygon_rate_limiter.wait_if_needed()
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={settings.POLYGON_API_KEY}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        
        if data["status"] != "OK" or data["resultsCount"] == 0:
            raise ValueError("Stock not found")
        
        price = float(data["results"][0]["c"])
        return {
            "symbol": symbol,
            "price": price,
            "last_updated": datetime.utcnow()
        }

async def get_crypto_price(symbol: str) -> Dict[str, Any]:
    """Get current cryptocurrency price using Polygon.io API"""
    await polygon_rate_limiter.wait_if_needed()
    
    url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/prev?apiKey={settings.POLYGON_API_KEY}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        
        if data["status"] != "OK" or data["resultsCount"] == 0:
            raise ValueError("Cryptocurrency not found")
        
        price = float(data["results"][0]["c"])
        return {
            "symbol": symbol,
            "price": price,
            "last_updated": datetime.now(timezone.utc)
        }
