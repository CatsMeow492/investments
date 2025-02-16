from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Dict, Any
from ..core.config import settings
from ..models.portfolio import ResearchQuery, ModelInfo
from ..services.ai import AIModel, query_ai_model, stream_ai_response
from ..services.market_data import fetch_web_data
from database import SessionLocal
from ..models.database import Portfolio, PortfolioAsset

router = APIRouter()

@router.get("/models")
async def get_available_models():
    """Get list of available AI models"""
    return AIModel.get_available_models()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/query")
async def research_query(
    query: ResearchQuery,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Process a research query using AI models"""
    try:
        # Verify at least one AI model is configured
        if not any([settings.OPENAI_API_KEY, settings.ANTHROPIC_API_KEY, settings.OPENROUTER_API_KEY]):
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
                portfolio = db.query(Portfolio).first()
                if not portfolio:
                    print("Warning: No portfolio found in database")
                else:
                    assets = db.query(PortfolioAsset).filter(
                        PortfolioAsset.portfolio_id == portfolio.id
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
            asset = db.query(PortfolioAsset).filter(
                PortfolioAsset.symbol == query.context.symbol
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
