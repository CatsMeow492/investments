from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    portfolios = relationship("Portfolio", back_populates="owner")

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="portfolios")
    assets = relationship("PortfolioAsset", back_populates="portfolio")

class PortfolioAsset(Base):
    __tablename__ = "portfolio_assets"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol = Column(String, index=True)
    name = Column(String)
    asset_type = Column(String)  # 'stock' or 'crypto'
    quantity = Column(Float)
    purchase_price = Column(Float)
    purchase_date = Column(DateTime)
    current_price = Column(Float, nullable=True)
    last_updated = Column(DateTime, nullable=True)

    portfolio = relationship("Portfolio", back_populates="assets")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_asset_id = Column(Integer, ForeignKey("portfolio_assets.id"))
    transaction_type = Column(String)  # 'buy' or 'sell'
    quantity = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    notes = Column(String, nullable=True) 