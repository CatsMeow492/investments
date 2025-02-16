from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .core.cache import connect_to_redis, close_redis
from .api import portfolio, research
from database import create_tables

# Create FastAPI app
app = FastAPI(title="Investment Portfolio Tracker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
create_tables()

# Register startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await connect_to_redis()

@app.on_event("shutdown")
async def shutdown_event():
    await close_redis()

# Import and include routers
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(research.router, prefix="/api/research", tags=["research"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Investment Portfolio Tracker API"} 