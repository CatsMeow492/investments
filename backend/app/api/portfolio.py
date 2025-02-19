from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from ..core.config import settings
from ..core.cache import get_cached_data, set_cached_data
from ..models.portfolio import PortfolioSummary, PortfolioAssetsResponse, AssetAllocation
from ..services.market_data import get_stock_price, get_crypto_price, fetch_web_data
from ..services.ai import query_ai_model
from database import SessionLocal
from ..models.database import Portfolio, PortfolioAsset
import asyncio
import logging

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(db: Session = Depends(get_db)):
    """Get portfolio summary including total value, gain/loss, and return"""
    portfolio = db.query(Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    assets = db.query(PortfolioAsset).filter(
        PortfolioAsset.portfolio_id == portfolio.id
    ).all()
    
    total_value = sum(asset.quantity * asset.current_price for asset in assets)
    total_cost = sum(asset.quantity * asset.purchase_price for asset in assets)
    total_gain_loss = total_value - total_cost
    gain_loss_percentage = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
    
    return PortfolioSummary(
        total_value=total_value,
        total_gain_loss=total_gain_loss,
        gain_loss_percentage=gain_loss_percentage,
        last_updated=datetime.utcnow()
    )

@router.get("/assets", response_model=PortfolioAssetsResponse)
async def get_portfolio_assets(db: Session = Depends(get_db)):
    """Get all assets in the portfolio"""
    try:
        # Try to get from cache first
        cache_key = "portfolio_assets"
        cached_assets = await get_cached_data(cache_key, "MARKET_DATA")
        
        if cached_assets:
            # Ensure update_status is included in cached response
            if "update_status" not in cached_assets:
                current_time = datetime.now(timezone.utc)
                next_update = current_time + timedelta(hours=24)
                cached_assets["update_status"] = {
                    "total_assets": len(cached_assets["assets"]),
                    "assets_needing_update": 0,
                    "next_update_attempt": next_update.isoformat()
                }
            print("Using cached portfolio assets")
            return PortfolioAssetsResponse(**cached_assets)
        
        portfolio = db.query(Portfolio).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        assets = db.query(PortfolioAsset).filter(
            PortfolioAsset.portfolio_id == portfolio.id
        ).all()
        
        current_time = datetime.now(timezone.utc)
        next_update = current_time + timedelta(hours=24)  # Schedule next update in 24 hours
        
        # Count assets needing update - ensure all datetimes are timezone-aware
        assets_needing_update = sum(
            1 for asset in assets
            if not asset.last_updated or 
            (asset.last_updated.replace(tzinfo=None) < (current_time - timedelta(hours=24)).replace(tzinfo=None))
        )
        
        response_data = {
            "assets": [{
                "symbol": asset.symbol,
                "name": asset.name,
                "type": asset.asset_type,
                "quantity": asset.quantity,
                "current_price": asset.current_price,
                "purchase_price": asset.purchase_price,
                "current_value": asset.quantity * asset.current_price,
                "purchase_value": asset.quantity * asset.purchase_price,
                "gain_loss": (asset.current_price - asset.purchase_price) * asset.quantity,
                "gain_loss_percentage": ((asset.current_price - asset.purchase_price) / asset.purchase_price * 100) 
                    if asset.purchase_price > 0 else 0,
                "last_updated": asset.last_updated.replace(tzinfo=timezone.utc).isoformat() if asset.last_updated else current_time.isoformat()
            } for asset in assets],
            "last_updated": current_time.isoformat(),
            "update_status": {
                "total_assets": len(assets),
                "assets_needing_update": assets_needing_update,
                "next_update_attempt": next_update.isoformat()
            }
        }
        
        # Cache the response
        await set_cached_data(cache_key, response_data, "MARKET_DATA")
        
        return PortfolioAssetsResponse(**response_data)
        
    except Exception as e:
        print(f"Error in get_portfolio_assets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/allocation", response_model=AssetAllocation)
async def get_portfolio_allocation(db: Session = Depends(get_db)):
    """Get portfolio allocation by asset type and individual holdings"""
    try:
        portfolio = db.query(Portfolio).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        assets = db.query(PortfolioAsset).filter(
            PortfolioAsset.portfolio_id == portfolio.id
        ).all()
        
        total_value = sum(asset.quantity * asset.current_price for asset in assets)
        
        # Group by asset type
        type_allocation = {}
        for asset in assets:
            value = asset.quantity * asset.current_price
            if asset.asset_type not in type_allocation:
                type_allocation[asset.asset_type] = 0
            type_allocation[asset.asset_type] += value
        
        # Calculate percentages for type allocation
        type_allocation_percentages = {
            asset_type: (value / total_value * 100) if total_value > 0 else 0
            for asset_type, value in type_allocation.items()
        }
        
        # Calculate individual asset allocation
        holdings_allocation = []
        for asset in assets:
            value = asset.quantity * asset.current_price
            percentage = (value / total_value * 100) if total_value > 0 else 0
            holdings_allocation.append({
                "symbol": asset.symbol,
                "name": asset.name,
                "value": value,
                "percentage": percentage
            })
        
        # Sort holdings by percentage in descending order
        holdings_allocation.sort(key=lambda x: x["percentage"], reverse=True)
        
        return AssetAllocation(
            total_value=total_value,
            by_type=type_allocation_percentages,
            by_holding=holdings_allocation
        )
    except Exception as e:
        print(f"Error in get_portfolio_allocation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/update-prices")
async def update_portfolio_prices(db: Session = Depends(get_db)):
    """Update current prices for all assets in the portfolio"""
    try:
        assets = db.query(PortfolioAsset).all()
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
                        price_data = await get_stock_price(asset.symbol)
                    else:  # crypto
                        price_data = await get_crypto_price(asset.symbol)
                    
                    asset.current_price = price_data["price"]
                    asset.last_updated = current_time
                    db.add(asset)
                    updated_assets.add(asset)
                            
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
        updated_assets.update(up_to_date_assets)
        
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

@router.get("/recommendations")
async def get_portfolio_recommendations(db: Session = Depends(get_db)):
    """Get AI-powered recommendations for portfolio holdings with caching"""
    try:
        # Try to get recommendations from cache first
        cache_key = "portfolio_recommendations"
        cached_recommendations = await get_cached_data(cache_key, "PORTFOLIO_RECOMMENDATIONS")
        
        if cached_recommendations:
            print("Using cached portfolio recommendations")
            return cached_recommendations
        
        assets = db.query(PortfolioAsset).all()
        recommendations = []
        
        # Calculate portfolio total value for percentage calculations
        total_portfolio_value = sum(asset.quantity * asset.current_price for asset in assets)
        
        # Process assets in smaller batches to avoid overwhelming APIs
        BATCH_SIZE = 3  # Process 3 assets at a time
        for i in range(0, len(assets), BATCH_SIZE):
            batch = assets[i:i + BATCH_SIZE]
            
            for asset in batch:
                try:
                    print(f"Processing recommendations for {asset.symbol}...")
                    web_data = await fetch_web_data(asset.symbol)
                    
                    # Validate that we have necessary data before proceeding
                    news_articles = web_data.get("news", [])
                    if not news_articles:
                        logger.warning(f"No news data available for {asset.symbol}, checking cache")
                        news_articles = await get_cached_data(f"news_{asset.symbol}", "NEWS")
                    
                    if not news_articles:
                        logger.warning(f"No news data available for {asset.symbol} in cache either, skipping AI analysis")
                        value = asset.quantity * asset.current_price
                        gain_loss = (asset.current_price - asset.purchase_price) * asset.quantity
                        gain_loss_percentage = ((asset.current_price - asset.purchase_price) / asset.purchase_price * 100) if asset.purchase_price > 0 else 0
                        portfolio_percentage = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                        
                        recommendations.append({
                            "symbol": asset.symbol,
                            "name": asset.name,
                            "value": value,
                            "analysis": (
                                f"Waiting for news data to provide analysis.\n"
                                f"Current Position:\n"
                                f"• Holdings: {asset.quantity:,.2f} shares at ${asset.current_price:,.2f}\n"
                                f"• Total Value: ${value:,.2f} ({portfolio_percentage:.1f}% of portfolio)\n"
                                f"• Performance: {gain_loss_percentage:+.2f}% (${gain_loss:+,.2f})"
                            ),
                            "news": [],
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        })
                        continue

                    # Calculate basic metrics
                    value = asset.quantity * asset.current_price
                    gain_loss = (asset.current_price - asset.purchase_price) * asset.quantity
                    gain_loss_percentage = ((asset.current_price - asset.purchase_price) / asset.purchase_price * 100) if asset.purchase_price > 0 else 0
                    portfolio_percentage = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                    
                    # Prepare context for AI analysis
                    analysis_context = {
                        "asset": {
                            "symbol": asset.symbol,
                            "name": asset.name,
                            "type": asset.asset_type,
                            "quantity": asset.quantity,
                            "current_price": asset.current_price,
                            "purchase_price": asset.purchase_price,
                            "current_value": value,
                            "portfolio_weight": portfolio_percentage,
                            "gain_loss": gain_loss,
                            "gain_loss_percentage": gain_loss_percentage
                        },
                        "market_data": web_data.get("market_data", {}),
                        "analyst_ratings": web_data.get("analyst_ratings", {}),
                        "news": news_articles
                    }
                    
                    # Generate AI analysis only if we have news data
                    analysis_prompt = f"""Analyze {asset.symbol} ({asset.name}) and determine if the outlook is BULLISH or BEARISH based on recent news and performance data.

Recent News Analysis:
{chr(10).join([f"• {article.get('title')} - {article.get('summary', '')[:200]}..." for article in news_articles[:3]])}

Position & Performance Data:
• Holdings: {asset.quantity:,.2f} shares at ${asset.current_price:,.2f}
• Total Value: ${value:,.2f} ({portfolio_percentage:.1f}% of portfolio)
• Performance: {gain_loss_percentage:+.2f}% (${gain_loss:+,.2f})
• Market Data: {web_data.get('market_data', {})}
• Analyst Ratings: {web_data.get('analyst_ratings', {})}

Provide a concise 3-4 sentence analysis that:
1. Starts with either "BULLISH" or "BEARISH" in caps
2. Directly references the most significant news item and its impact
3. Integrates performance data with news sentiment
4. Concludes with a specific recommendation for position management"""

                    try:
                        ai_response = await query_ai_model(analysis_prompt, analysis_context)
                        ai_analysis = ai_response.get("answer", "Analysis pending...")
                    except Exception as ai_error:
                        print(f"Error generating AI analysis for {asset.symbol}: {str(ai_error)}")
                        ai_analysis = f"Error generating analysis. Current Position: {asset.quantity:,.2f} shares at ${asset.current_price:,.2f}, total value ${value:,.2f} ({portfolio_percentage:.1f}% of portfolio). Performance: {gain_loss_percentage:+.2f}% (${gain_loss:+,.2f})"
                    
                    # Create recommendation with all available data
                    recommendation = {
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "value": value,
                        "analysis": ai_analysis,
                        "news": news_articles[:3],  # Include top 3 news articles
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                    
                    recommendations.append(recommendation)
                    
                    # Add a smaller delay between individual asset processing
                    await asyncio.sleep(1)
    
                except Exception as e:
                    print(f"Error analyzing {asset.symbol}: {str(e)}")
                    # Still provide basic information even if web data fails
                    value = asset.quantity * asset.current_price
                    gain_loss = (asset.current_price - asset.purchase_price) * asset.quantity
                    gain_loss_percentage = ((asset.current_price - asset.purchase_price) / asset.purchase_price * 100) if asset.purchase_price > 0 else 0
                    portfolio_percentage = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                    
                    recommendations.append({
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "value": value,
                        "analysis": (
                            f"Position Summary:\n"
                            f"• Holdings: {asset.quantity:,.2f} shares at ${asset.current_price:,.2f}\n"
                            f"• Total Value: ${value:,.2f} ({portfolio_percentage:.1f}% of portfolio)\n"
                            f"• Performance: {gain_loss_percentage:+.2f}% (${gain_loss:+,.2f})\n\n"
                            "Additional market data temporarily unavailable."
                        ),
                        "news": [],
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    })
                
            # Add a smaller delay between batches
            if i + BATCH_SIZE < len(assets):
                print("Waiting between batches to respect API rate limits...")
                await asyncio.sleep(3)
        
        response_data = {
            "recommendations": recommendations,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "error_count": len([r for r in recommendations if not r.get("news")])
        }
        
        # Cache the recommendations
        await set_cached_data(cache_key, response_data, "PORTFOLIO_RECOMMENDATIONS")
        
        return response_data
        
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
