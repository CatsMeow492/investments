from datetime import datetime, timezone
from database import SessionLocal, create_tables
from app.models.database import Portfolio, PortfolioAsset

def init_db():
    # Create tables
    create_tables()
    
    # Create a database session
    db = SessionLocal()
    
    try:
        # Check if we already have a portfolio
        existing_portfolio = db.query(Portfolio).first()
        if existing_portfolio:
            print("Database already initialized")
            return
        
        # Create the actual portfolio
        portfolio = Portfolio(
            name="My Investment Portfolio",
            description="A diversified portfolio of ETFs and growth stocks"
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
        
        # Actual portfolio assets
        portfolio_assets = [
            {
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF TRUST",
                "asset_type": "stock",
                "quantity": 65.0172,
                "purchase_price": 603.36,
                "current_price": 609.70,
                "purchase_date": datetime.now(timezone.utc)
            },
            {
                "symbol": "VOO",
                "name": "VANGUARD S&P 500 ETF",
                "asset_type": "stock",
                "quantity": 40.0566,
                "purchase_price": 554.81,
                "current_price": 560.69,
                "purchase_date": datetime.now(timezone.utc)
            },
            {
                "symbol": "TEM",
                "name": "TEMPUS AI INC CLASS A COMMON STOCK",
                "asset_type": "stock",
                "quantity": 200.0000,
                "purchase_price": 73.88,
                "current_price": 89.44,
                "purchase_date": datetime.now(timezone.utc)
            },
            {
                "symbol": "QQQ",
                "name": "INVESCO QQQ TR UNIT SER 1",
                "asset_type": "stock",
                "quantity": 24.0608,
                "purchase_price": 528.30,
                "current_price": 528.30,
                "purchase_date": datetime.now(timezone.utc)
            },
            {
                "symbol": "MSFT",
                "name": "MICROSOFT CORP",
                "asset_type": "stock",
                "quantity": 27.5060,
                "purchase_price": 409.04,
                "current_price": 409.04,
                "purchase_date": datetime.now(timezone.utc)
            },
            {
                "symbol": "PLTR",
                "name": "PALANTIR TECHNOLOGIES INC",
                "asset_type": "stock",
                "quantity": 66.1696,
                "purchase_price": 117.39,
                "current_price": 117.39,
                "purchase_date": datetime.now(timezone.utc)
            },
            {
                "symbol": "LUNR",
                "name": "INTUITIVE MACHINES INC",
                "asset_type": "stock",
                "quantity": 410.0000,
                "purchase_price": 18.75,
                "current_price": 18.75,
                "purchase_date": datetime.now(timezone.utc)
            }
        ]
        
        # Add assets to portfolio
        for asset_data in portfolio_assets:
            asset = PortfolioAsset(
                portfolio_id=portfolio.id,
                **asset_data
            )
            db.add(asset)
        
        db.commit()
        print("Database initialized with actual portfolio data")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db() 