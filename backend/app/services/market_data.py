import httpx
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import os
from ..core.config import settings
from ..core.cache import get_cached_data, set_cached_data, clear_cache
import aiohttp
import logging
from bs4 import BeautifulSoup
import re
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        logger.info(f"Initializing Polygon client with base URL: {self.base_url}")
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def get_last_trade(self, symbol: str):
        """Get the last trade for a symbol"""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        logger.info(f"Fetching last trade for {symbol} from Polygon")
        return await self.session.get(url)

    async def get_analyst_ratings(self, symbol: str):
        """Get analyst ratings for a symbol"""
        url = f"{self.base_url}/v3/snapshot/analyst-ratings/stocks/{symbol}"
        logger.info(f"Fetching analyst ratings for {symbol} from Polygon")
        return await self.session.get(url)

    async def close(self):
        """Close the client session"""
        logger.info("Closing Polygon client session")
        await self.session.close()

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

# Create global instances
logger.info("Creating global Polygon instances...")
polygon_rate_limiter = PolygonRateLimiter(calls_per_minute=settings.POLYGON_CALLS_PER_MINUTE)
polygon_client = PolygonClient(settings.POLYGON_API_KEY)

# Initialize Selenium WebDriver with options
def get_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

async def scrape_yahoo_finance_news(symbol: str) -> list:
    """Scrape news from Yahoo Finance using Selenium for dynamic content"""
    driver = None
    try:
        logger.info(f"Attempting to scrape Yahoo Finance news for {symbol}")
        encoded_symbol = quote(symbol)
        url = f"https://finance.yahoo.com/quote/{encoded_symbol}"
        
        # Initialize the driver with updated options
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # Use new headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-web-security")  # Disable CORS
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation
        chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set page load timeout
        driver.set_page_load_timeout(20)
        
        logger.info(f"Navigating to {url}")
        driver.get(url)
        
        # Log the page title to verify we're on the right page
        logger.info(f"Page title: {driver.title}")
        
        # Wait for the page to load
        wait = WebDriverWait(driver, 20)  # Increased timeout to 20 seconds
        
        # Accept any consent dialogs if present
        try:
            consent_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'Agree')]"))
            )
            consent_button.click()
            logger.info("Clicked consent button")
        except TimeoutException:
            logger.info("No consent button found")
        
        # Try to scroll to load more content
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for content to load
            driver.execute_script("window.scrollTo(0, 0);")  # Scroll back to top
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Error during scrolling: {str(e)}")
        
        # Log the current page source for debugging
        logger.info("Current page source structure:")
        logger.info(driver.page_source[:1000])  # Log first 1000 characters
        
        # Updated selectors based on the provided Yahoo Finance structure
        news_items = []
        selectors = [
            "#news-06rzpbhv > div.filtered-stories.small.yf-186c5b2.rulesBetween article",  # Main news container
            "#news-06rzpbhv article",  # Fallback for main container
            "div.filtered-stories article",  # Generic filtered stories
            "div.yf-186c5b2 article"  # Class-based selector
        ]
        
        for selector in selectors:
            try:
                logger.info(f"Trying selector: {selector}")
                
                # Wait for elements and get them using CSS selector
                news_elements = wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                )
                
                logger.info(f"Found {len(news_elements)} news elements with selector '{selector}' for {symbol}")
                
                if news_elements:
                    # Log the HTML of the first element for debugging
                    if news_elements[0]:
                        logger.info(f"First element HTML: {news_elements[0].get_attribute('outerHTML')}")
                    
                    for element in news_elements[:5]:  # Get top 5 news items
                        try:
                            # Get the title and URL from the article
                            title_element = element.find_element(By.CSS_SELECTOR, "h3 a, a.title")
                            title = title_element.text.strip()
                            url = title_element.get_attribute("href")
                            
                            # Try to get the summary
                            try:
                                summary_element = element.find_element(By.CSS_SELECTOR, "p.summary, div.summary")
                                summary = summary_element.text.strip()
                            except NoSuchElementException:
                                summary = ""
                            
                            if title and url:
                                logger.info(f"Found news item - Title: {title[:50]}...")
                                news_items.append({
                                    "title": title,
                                    "url": url,
                                    "summary": summary,
                                    "published_at": datetime.now(timezone.utc).isoformat(),
                                    "source": "Yahoo Finance"
                                })
                        
                        except Exception as e:
                            logger.error(f"Error parsing news item for {symbol}: {str(e)}")
                            continue
                    
                    if news_items:
                        logger.info(f"Successfully found {len(news_items)} news items using selector '{selector}'")
                        break  # Stop if we found news items with current selector
                
            except TimeoutException:
                logger.warning(f"Timeout waiting for selector {selector} for {symbol}")
                continue
            except Exception as e:
                logger.error(f"Error with selector {selector}: {str(e)}")
                continue
        
        logger.info(f"Successfully scraped {len(news_items)} news items from Yahoo Finance for {symbol}")
        return news_items
        
    except Exception as e:
        logger.error(f"Error scraping Yahoo Finance news for {symbol}: {str(e)}")
        return []
        
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                logger.error(f"Error closing Selenium driver: {str(e)}")

async def get_alpha_vantage_news(symbol: str) -> list:
    """Get news from Alpha Vantage API with rate limiting"""
    if not settings.ALPHA_VANTAGE_API_KEY:
        return []
    
    try:
        logger.info(f"Fetching news from Alpha Vantage for {symbol}")
        news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={settings.ALPHA_VANTAGE_API_KEY}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(news_url) as response:
                if response.status == 200:
                    news_data = await response.json()
                    if "feed" in news_data and news_data["feed"]:
                        news = [
                            {
                                "title": article.get("title", ""),
                                "url": article.get("url", ""),
                                "published_at": article.get("time_published", ""),
                                "summary": article.get("summary", ""),
                                "source": "Alpha Vantage",
                                "sentiment": article.get("overall_sentiment_score", 0)
                            }
                            for article in news_data["feed"][:5]
                        ]
                        logger.info(f"Successfully fetched {len(news)} news articles from Alpha Vantage for {symbol}")
                        return news
                elif response.status == 429:
                    logger.warning(f"Rate limit hit for Alpha Vantage news - {symbol}")
                else:
                    logger.error(f"Alpha Vantage request failed with status {response.status}")
    except Exception as e:
        logger.error(f"Error fetching news from Alpha Vantage for {symbol}: {str(e)}")
    
    return []

async def aggregate_news(symbol: str) -> list:
    """Aggregate news from multiple sources asynchronously"""
    try:
        # Get news from all sources concurrently
        yahoo_news_task = asyncio.create_task(scrape_yahoo_finance_news(symbol))
        alpha_vantage_news_task = asyncio.create_task(get_alpha_vantage_news(symbol))
        
        # Wait for all news sources to complete
        results = await asyncio.gather(
            yahoo_news_task,
            alpha_vantage_news_task,
            return_exceptions=True
        )
        
        # Combine all news items
        all_news = []
        for result in results:
            if isinstance(result, list):  # Only add successful results
                all_news.extend(result)
        
        # Sort by published date if available, otherwise keep original order
        all_news.sort(
            key=lambda x: datetime.fromisoformat(x["published_at"]) if x.get("published_at") else datetime.min,
            reverse=True
        )
        
        # Remove duplicates based on title similarity
        unique_news = []
        seen_titles = set()
        for news in all_news:
            title = news["title"].lower()
            if not any(title in seen_title or seen_title in title for seen_title in seen_titles):
                seen_titles.add(title)
                unique_news.append(news)
        
        logger.info(f"Aggregated {len(unique_news)} unique news items for {symbol}")
        return unique_news[:5]  # Return top 5 most recent unique news items
        
    except Exception as e:
        logger.error(f"Error aggregating news for {symbol}: {str(e)}")
        return []

async def fetch_web_data(symbol: str) -> dict:
    """Fetch market data, news, and analyst ratings for a given symbol"""
    logger.info(f"Starting web data fetch for {symbol}")
    try:
        # Try to get from cache first
        cache_key = f"web_data_{symbol}"
        cached_data = await get_cached_data(cache_key, "MARKET_DATA")
        
        if cached_data and cached_data.get("news"):  # Only use cache if it has news
            logger.info(f"Using cached web data for {symbol}")
            return cached_data
        
        # Start news aggregation early since it's independent
        logger.info(f"Starting news aggregation for {symbol}")
        news_task = asyncio.create_task(aggregate_news(symbol))
        
        market_data = {}
        analyst_ratings = {}
        
        # Fetch market data from Polygon
        try:
            logger.info(f"Fetching market data for {symbol} from Polygon")
            await polygon_rate_limiter.wait_if_needed()
            response = await polygon_client.get_last_trade(symbol)
            
            if response.status == 200:
                trade_data = await response.json()
                logger.info(f"Received market data response for {symbol}")
                if "results" in trade_data:
                    result = trade_data["results"]
                    market_data = {
                        "price": result.get("p", 0),
                        "volume": result.get("v", 0),
                        "open": result.get("o", 0),
                        "high": result.get("h", 0),
                        "low": result.get("l", 0),
                        "timestamp": result.get("t", "")
                    }
            elif response.status == 403:
                logger.error(f"Authentication failed for Polygon API - please verify your API key is valid")
            elif response.status == 429:
                logger.warning(f"Rate limit hit for Polygon market data - {symbol}")
                market_data = await get_cached_data(f"market_data_{symbol}", "MARKET_DATA") or {}
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}", exc_info=True)
        
        # Fetch analyst ratings from Polygon
        try:
            logger.info(f"Fetching analyst ratings for {symbol} from Polygon")
            await polygon_rate_limiter.wait_if_needed()
            response = await polygon_client.get_analyst_ratings(symbol)
            
            if response.status == 200:
                ratings_data = await response.json()
                logger.info(f"Received analyst ratings response for {symbol}")
                if "results" in ratings_data:
                    result = ratings_data["results"]
                    analyst_ratings = {
                        "price_target": result.get("price_target", {}).get("average", 0),
                        "recommendations": result.get("recommendation", {}).get("text", ""),
                        "updated_at": result.get("updated_at", "")
                    }
            elif response.status == 429:
                logger.warning(f"Rate limit hit for Polygon analyst ratings - {symbol}")
                analyst_ratings = await get_cached_data(f"ratings_{symbol}", "RATINGS") or {}
        except Exception as e:
            logger.error(f"Error fetching analyst ratings for {symbol}: {str(e)}", exc_info=True)
        
        # Wait for news aggregation to complete
        logger.info(f"Waiting for news aggregation to complete for {symbol}")
        news = await news_task
        logger.info(f"News aggregation completed for {symbol}. Got {len(news)} articles")
        
        # Only cache if we have actual news
        if news:
            logger.info(f"Caching {len(news)} news articles for {symbol}")
            await set_cached_data(f"news_{symbol}", news, "NEWS")
        else:
            logger.warning(f"No news articles found for {symbol}, skipping cache")
            # Clear any existing cached news to prevent serving stale data
            await clear_cache(f"news_{symbol}")
        
        # Combine all data
        web_data = {
            "market_data": market_data,
            "news": news,
            "analyst_ratings": analyst_ratings
        }
        
        # Only cache the combined data if we have news
        if news:
            await set_cached_data(cache_key, web_data, "MARKET_DATA")
            logger.info(f"Cached combined web data for {symbol}")
        
        return web_data
        
    except Exception as e:
        logger.error(f"Error in fetch_web_data for {symbol}: {str(e)}", exc_info=True)
        return {
            "market_data": {},
            "news": [],
            "analyst_ratings": {}
        }

async def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get current stock price using Polygon.io API"""
    logger.info(f"Getting stock price for {symbol}")
    await polygon_rate_limiter.wait_if_needed()
    
    try:
        response = await polygon_client.get_last_trade(symbol)
        if response.status == 200:
            data = await response.json()
            logger.info(f"Received price data for {symbol}: {data}")
            if "results" in data:
                result = data["results"]
                return {
                    "symbol": symbol,
                    "price": result.get("p", 0),
                    "last_updated": datetime.now(timezone.utc)
                }
        logger.error(f"Stock data not found for {symbol}")
        raise ValueError("Stock data not found")
    except Exception as e:
        logger.error(f"Error getting stock price for {symbol}: {str(e)}", exc_info=True)
        raise

async def get_crypto_price(symbol: str) -> Dict[str, Any]:
    """Get current cryptocurrency price using Polygon.io API"""
    logger.info(f"Getting crypto price for {symbol}")
    await polygon_rate_limiter.wait_if_needed()
    
    try:
        response = await polygon_client.get_last_trade(f"X:{symbol}USD")
        if response.status == 200:
            data = await response.json()
            logger.info(f"Received crypto price data for {symbol}: {data}")
            if "results" in data:
                result = data["results"]
                return {
                    "symbol": symbol,
                    "price": result.get("p", 0),
                    "last_updated": datetime.now(timezone.utc)
                }
        logger.error(f"Cryptocurrency data not found for {symbol}")
        raise ValueError("Cryptocurrency data not found")
    except Exception as e:
        logger.error(f"Error getting crypto price for {symbol}: {str(e)}", exc_info=True)
        raise

# Cleanup function for the Polygon client
async def cleanup():
    """Cleanup function to close the Polygon client session"""
    logger.info("Cleaning up Polygon client")
    await polygon_client.close()
