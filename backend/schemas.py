from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    
    class Config:
        from_attributes = True

# Portfolio schemas
class PortfolioBase(BaseModel):
    name: str
    description: Optional[str] = None

class PortfolioCreate(PortfolioBase):
    pass

class Portfolio(PortfolioBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Asset schemas
class AssetBase(BaseModel):
    symbol: str
    name: str
    asset_type: str
    quantity: float
    purchase_price: float
    purchase_date: datetime

class AssetCreate(AssetBase):
    portfolio_id: int

class Asset(AssetBase):
    id: int
    current_price: Optional[float] = None
    last_updated: Optional[datetime] = None
    portfolio_id: int

    class Config:
        from_attributes = True

# Transaction schemas
class TransactionBase(BaseModel):
    transaction_type: str
    quantity: float
    price: float
    notes: Optional[str] = None

class TransactionCreate(TransactionBase):
    portfolio_asset_id: int

class Transaction(TransactionBase):
    id: int
    portfolio_asset_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# Portfolio Performance
class PortfolioPerformance(BaseModel):
    total_value: float
    total_cost: float
    total_gain_loss: float
    gain_loss_percentage: float
    last_updated: datetime

# Asset Price Update
class AssetPriceUpdate(BaseModel):
    current_price: float
    last_updated: datetime 