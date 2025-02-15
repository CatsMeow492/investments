# Investment Portfolio Tracker

A modern web application for tracking, researching, and visualizing stock and cryptocurrency investments.

## Features

- Track stocks and cryptocurrencies in real-time
- Visualize portfolio performance with interactive charts
- Research assets with historical data and key metrics
- Monitor portfolio allocation and returns
- Real-time price updates

## Tech Stack

- Backend: Python with FastAPI
- Frontend: React with TypeScript
- Database: SQLite
- Data Visualization: Recharts
- APIs: 
  - Alpha Vantage (Stocks)
  - CoinGecko (Cryptocurrencies)

## Setup Instructions

1. Clone the repository
2. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```
4. Create a `.env` file in the backend directory with your API keys:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

## Running the Application

1. Start the backend:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm start
   ```

The application will be available at http://localhost:3000 # investments
