from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# Request/Response Models
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

# Response Models
class PortfolioSummary(BaseModel):
    total_value: float
    total_gain_loss: float
    gain_loss_percentage: float
    last_updated: datetime

class AssetAllocation(BaseModel):
    total_value: float
    by_type: dict
    by_holding: List[dict]

class PortfolioAssetResponse(BaseModel):
    symbol: str
    name: str
    type: str
    quantity: float
    current_price: float
    purchase_price: float
    current_value: float
    purchase_value: float
    gain_loss: float
    gain_loss_percentage: float
    last_updated: Optional[str]

class UpdateStatus(BaseModel):
    total_assets: int
    assets_needing_update: int
    next_update_attempt: str

class PortfolioAssetsResponse(BaseModel):
    assets: List[PortfolioAssetResponse]
    last_updated: str
    update_status: UpdateStatus  # Add update status to match frontend expectations
