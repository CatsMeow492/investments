from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from datetime import datetime

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

def import_portfolio():
    db = SessionLocal()
    try:
        # Create a default user if not exists
        user = db.query(models.User).first()
        if not user:
            user = models.User(
                email="default@example.com",
                hashed_password="default",  # In production, this should be properly hashed
                full_name="Default User"
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # Create a main portfolio
        portfolio = models.Portfolio(
            name="Main Portfolio",
            description="Imported portfolio",
            user_id=user.id
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)

        # Investment data
        investments = [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF TRUST", "quantity": 65.01718, "current_price": 603.36, "asset_type": "stock"},
            {"symbol": "VOO", "name": "VANGUARD S&P 500 ETF", "quantity": 40.05663, "current_price": 554.81, "asset_type": "stock"},
            {"symbol": "TEM", "name": "TEMPUS AI INC CLASS A COMMON STOCK", "quantity": 200, "current_price": 73.88, "asset_type": "stock"},
            {"symbol": "QQQ", "name": "INVESCO QQQ TR UNIT SER 1", "quantity": 24.06079, "current_price": 528.30, "asset_type": "stock"},
            {"symbol": "MSFT", "name": "MICROSOFT CORP", "quantity": 27.50596, "current_price": 409.04, "asset_type": "stock"},
            {"symbol": "PLTR", "name": "PALANTIR TECHNOLOGIES INC", "quantity": 66.16957, "current_price": 117.39, "asset_type": "stock"},
            {"symbol": "LUNR", "name": "INTUITIVE MACHINES INC", "quantity": 410, "current_price": 18.75, "asset_type": "stock"},
            {"symbol": "SMR", "name": "NUSCALE POWER CORPORATION", "quantity": 211.41966, "current_price": 26.40, "asset_type": "stock"},
            {"symbol": "AMD", "name": "ADVANCED MICRO DEVICES INC", "quantity": 42.66292, "current_price": 111.72, "asset_type": "stock"},
            {"symbol": "AAAU", "name": "GOLDMAN SACHS PHYSICAL GOLD ETF", "quantity": 139.01731, "current_price": 28.685, "asset_type": "stock"},
            {"symbol": "CRSP", "name": "CRISPR THERAPEUTICS AG", "quantity": 89.33444, "current_price": 43.30, "asset_type": "stock"},
            {"symbol": "RGTI", "name": "RIGETTI COMPUTING INC", "quantity": 302, "current_price": 11.75, "asset_type": "stock"},
            {"symbol": "NVDA", "name": "NVIDIA CORP", "quantity": 26.49772, "current_price": 131.14, "asset_type": "stock"},
            {"symbol": "RKLB", "name": "ROCKET LAB USA INC", "quantity": 119.53442, "current_price": 27.62, "asset_type": "stock"},
            {"symbol": "CCJ", "name": "CAMECO CORP", "quantity": 59.99863, "current_price": 49.63, "asset_type": "stock"},
            {"symbol": "ISRG", "name": "INTUITIVE SURGICAL INC", "quantity": 5, "current_price": 589.61, "asset_type": "stock"},
            {"symbol": "COST", "name": "COSTCO WHOLESALE CORP", "quantity": 2.68162, "current_price": 1065.12, "asset_type": "stock"},
            {"symbol": "XLF", "name": "FINANCIAL SELECT SECTOR SPDR", "quantity": 50.2113, "current_price": 51.36, "asset_type": "stock"},
            {"symbol": "QUBT", "name": "QUANTUM COMPUTING INC", "quantity": 300, "current_price": 8.26, "asset_type": "stock"},
            {"symbol": "PANW", "name": "PALO ALTO NETWORKS INC", "quantity": 12, "current_price": 196.73, "asset_type": "stock"},
            {"symbol": "IOVA", "name": "IOVANCE BIOTHERAPEUTICS INC", "quantity": 376.03578, "current_price": 5.34, "asset_type": "stock"},
            {"symbol": "NBIS", "name": "NEBIUS GROUP N V", "quantity": 50, "current_price": 39.30, "asset_type": "stock"},
            {"symbol": "GOOGL", "name": "ALPHABET INC CLASS A", "quantity": 10.51963, "current_price": 183.61, "asset_type": "stock"},
            {"symbol": "SHOC", "name": "STRIVE U S SEMICONDUCTOR ETF", "quantity": 40.9155, "current_price": 47.1115, "asset_type": "stock"},
            {"symbol": "CRWD", "name": "CROWDSTRIKE HOLDINGS INC", "quantity": 4.33305, "current_price": 434.63, "asset_type": "stock"},
            {"symbol": "QBTS", "name": "D WAVE QUANTUM INC", "quantity": 300, "current_price": 6.04, "asset_type": "stock"},
            {"symbol": "INTC", "name": "INTEL CORP", "quantity": 80.43343, "current_price": 22.48, "asset_type": "stock"},
            {"symbol": "CEG", "name": "CONSTELLATION ENERGY CORPORATION", "quantity": 5.69809, "current_price": 313.80, "asset_type": "stock"},
            {"symbol": "SGOL", "name": "ABRDN PHYSICAL GOLD SHARES ETF", "quantity": 55, "current_price": 27.69, "asset_type": "stock"},
            {"symbol": "ENSG", "name": "ENSIGN GROUP INC", "quantity": 10, "current_price": 126.12, "asset_type": "stock"},
            {"symbol": "RXRX", "name": "RECURSION PHARMACEUTICALS INC", "quantity": 150, "current_price": 8.34, "asset_type": "stock"},
            {"symbol": "DNA", "name": "GINKGO BIOWORKS HOLDINGS INC", "quantity": 84.87783, "current_price": 12.32, "asset_type": "stock"},
            {"symbol": "UEC", "name": "URANIUM ENERGY CORP", "quantity": 144.55189, "current_price": 7.06, "asset_type": "stock"},
            {"symbol": "ARGT", "name": "GLOBAL X MSCI ARGENTINA ETF", "quantity": 11.56261, "current_price": 83.72, "asset_type": "stock"},
            {"symbol": "VIST", "name": "VISTA ENERGY S A B DE C V", "quantity": 17, "current_price": 50.71, "asset_type": "stock"},
            {"symbol": "RR", "name": "RICHTECH ROBOTICS INC", "quantity": 300, "current_price": 2.64, "asset_type": "stock"},
            {"symbol": "EWW", "name": "ISHARES MSCI MEXICO ETF", "quantity": 14.58596, "current_price": 51.91, "asset_type": "stock"},
            {"symbol": "ROBT", "name": "FIRST TR NASDAQ AI AND ROBOTICS ETF", "quantity": 15.07393, "current_price": 48.13, "asset_type": "stock"},
            {"symbol": "UUUU", "name": "ENERGY FUELS INC", "quantity": 142.14443, "current_price": 5.03, "asset_type": "stock"},
            {"symbol": "IGIC", "name": "INTERNATIONAL GENERAL INSURANCE", "quantity": 25, "current_price": 26.08, "asset_type": "stock"},
            {"symbol": "ISNPY", "name": "INTESA SANPAOLO SPA", "quantity": 20, "current_price": 27.7725, "asset_type": "stock"},
            {"symbol": "BBVA", "name": "BANCO BILBAO VIZCAYA ARGENTARIA", "quantity": 42.13681, "current_price": 12.43, "asset_type": "stock"},
            {"symbol": "VICI", "name": "VICI PROPERTIES INC", "quantity": 16.2366, "current_price": 29.79, "asset_type": "stock"},
            {"symbol": "EWI", "name": "ISHARES MSCI ITALY ETF", "quantity": 10, "current_price": 39.95, "asset_type": "stock"},
            {"symbol": "TLH", "name": "ISHARES 10-20 YEAR TREASURY BOND ETF", "quantity": 2.1541, "current_price": 99.53, "asset_type": "stock"},
            {"symbol": "BYDDY", "name": "BYD COMPANY LTD", "quantity": 2, "current_price": 91.565, "asset_type": "stock"},
            {"symbol": "GDX", "name": "VANECK GOLD MINERS ETF", "quantity": 2.60678, "current_price": 42.13, "asset_type": "stock"},
            {"symbol": "LI", "name": "LI AUTO INC", "quantity": 3.41863, "current_price": 26.30, "asset_type": "stock"},
            {"symbol": "TSLA", "name": "TESLA INC", "quantity": 0.1228, "current_price": 336.51, "asset_type": "stock"},
            {"symbol": "VRTX", "name": "VERTEX PHARMACEUTICALS INC", "quantity": 0.0905, "current_price": 453.20, "asset_type": "stock"},
            {"symbol": "XPEV", "name": "XPENG INC", "quantity": 1.94979, "current_price": 16.03, "asset_type": "stock"},
            {"symbol": "BMWKY", "name": "BAYERISCHE MOTOREN WERKE AG", "quantity": 1, "current_price": 26.626, "asset_type": "stock"},
            {"symbol": "AVGO", "name": "BROADCOM INC", "quantity": 0.09335, "current_price": 236.35, "asset_type": "stock"}
        ]

        # Add each investment to the portfolio
        for inv in investments:
            asset = models.PortfolioAsset(
                portfolio_id=portfolio.id,
                symbol=inv["symbol"],
                name=inv["name"],
                asset_type=inv["asset_type"],
                quantity=inv["quantity"],
                current_price=inv["current_price"],
                purchase_price=inv["current_price"],  # Using current price as purchase price since historical data not provided
                purchase_date=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            db.add(asset)

        db.commit()
        print("Portfolio imported successfully!")

    except Exception as e:
        print(f"Error importing portfolio: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import_portfolio() 