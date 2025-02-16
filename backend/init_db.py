from datetime import datetime
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
        
        # Create a sample portfolio
        portfolio = Portfolio(
            name="My Investment Portfolio",
            description="A diversified portfolio of stocks and cryptocurrencies"
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
        
        # Sample assets
        sample_assets = [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "asset_type": "stock",
                "quantity": 10,
                "purchase_price": 150.0,
                "current_price": 175.0,
                "purchase_date": datetime(2023, 1, 1)
            },
            {
                "symbol": "GOOGL",
                "name": "Alphabet Inc.",
                "asset_type": "stock",
                "quantity": 5,
                "purchase_price": 2000.0,
                "current_price": 2100.0,
                "purchase_date": datetime(2023, 2, 1)
            },
            {
                "symbol": "BTC",
                "name": "Bitcoin",
                "asset_type": "crypto",
                "quantity": 0.5,
                "purchase_price": 30000.0,
                "current_price": 35000.0,
                "purchase_date": datetime(2023, 3, 1)
            }
        ]
        
        # Add assets to portfolio
        for asset_data in sample_assets:
            asset = PortfolioAsset(
                portfolio_id=portfolio.id,
                **asset_data
            )
            db.add(asset)
        
        db.commit()
        print("Database initialized with sample data")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db() 