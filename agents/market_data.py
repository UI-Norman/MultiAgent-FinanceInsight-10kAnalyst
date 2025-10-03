from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
import yfinance as yf
from datetime import datetime
from core.citations import Citation
from typing import Dict
from phi.model.openai import OpenAIChat

class MarketDataAgent:
    """Fetch real-time market metrics"""
    
    def __init__(self):
        self.agent = Agent(
            name="Market Data Agent",
            model=OpenAIChat(id="gpt-4o"),  
            tools=[YFinanceTools(
                stock_price=True,
                company_info=True,
                analyst_recommendations=True,
                company_news=True
            )],
            instructions=[
                "Fetch latest price, market cap, shares outstanding",
                "Include timestamp and data source",
                "Return structured JSON with all metrics"
            ]
        )

    # Add run method that orchestrator calls:
    def run(self, ticker: str) -> Dict:
        """Fetch market data for ticker"""
        return self.get_comprehensive_data(ticker)

    def get_comprehensive_data(self, ticker: str) -> Dict:
        """Get all available market data"""
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        timestamp = datetime.now().isoformat()
        
        data = {
            "ticker": ticker,
            "timestamp": timestamp,
            "current_price": info.get('currentPrice'),
            "previous_close": info.get('previousClose'),
            "open": info.get('open'),
            "market_cap": info.get('marketCap'),
            "shares_outstanding": info.get('sharesOutstanding'),
            "52_week_high": info.get('fiftyTwoWeekHigh'),
            "52_week_low": info.get('fiftyTwoWeekLow'),
            "pe_ratio": info.get('trailingPE'),
            "forward_pe": info.get('forwardPE'),
            "dividend_yield": info.get('dividendYield'),
            "beta": info.get('beta'),
            "volume": info.get('volume'),
            "avg_volume": info.get('averageVolume'),
            "currency": info.get('currency', 'USD'),
            "exchange": info.get('exchange'),
            "timezone": info.get('timeZoneFullName'),
            "citation": Citation(
                source_type="market_data",
                ticker=ticker,
                year=None,
                section=None,
                url=f"https://finance.yahoo.com/quote/{ticker}",
                timestamp=timestamp
            )
        }
        
        return data
    
    def validate_data(self, data: Dict) -> Dict[str, bool]:
        """Validate market data integrity"""
        
        validations = {
            "has_price": data.get('current_price') is not None,
            "has_market_cap": data.get('market_cap') is not None,
            "has_shares": data.get('shares_outstanding') is not None,
            "market_cap_matches": self._verify_market_cap(data)
        }
        
        return validations
    
    def _verify_market_cap(self, data: Dict) -> bool:
        """Verify market_cap ≈ price × shares_outstanding"""
        
        price = data.get('current_price')
        shares = data.get('shares_outstanding')
        reported_cap = data.get('market_cap')
        
        if not all([price, shares, reported_cap]):
            return False
        
        calculated_cap = price * shares
        tolerance = 0.05  # 5% tolerance
        
        return abs(calculated_cap - reported_cap) / reported_cap < tolerance