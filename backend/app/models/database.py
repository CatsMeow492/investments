from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database import Base

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

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
    current_price = Column(Float)
    purchase_date = Column(DateTime)
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    portfolio = relationship("Portfolio", back_populates="assets") 