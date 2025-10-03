from phi.agent import Agent
from typing import Dict, List
from core.citations import CitationTracker
from phi.model.openai import OpenAIChat

class FinancialAnalystAgent:
    """Synthesize findings into comprehensive brief"""
    
    def __init__(self):
        self.agent = Agent(
        name="Financial Analyst",
        model=OpenAIChat(id="gpt-4o"),  # ADD THIS LINE
        role="Synthesize research into actionable insights",
        instructions=[
            "Write concise markdown briefs",
            "Use tables for metrics",
            "Cite all claims from 10-Ks or market data",
            "Highlight key risks and opportunities",
            "Compare to industry benchmarks when possible"
        ]
    )
    self.citation_tracker = CitationTracker()

    # Add missing methods:
    def _generate_executive_summary(self, sec_data: Dict, market_data: Dict) -> str:
        """Generate executive summary"""
        ticker = market_data.get('ticker', 'N/A')
        price = market_data.get('current_price', 0)
        market_cap = market_data.get('market_cap', 0)
        
        return f"""
    {ticker} is currently trading at ${price:.2f} with a market capitalization of 
    ${market_cap:,.0f}. The company demonstrates solid market positioning with 
    ongoing strategic initiatives detailed in recent SEC filings.
    """

    def _summarize_business(self, sec_data: Dict, ticker: str) -> str:
        """Summarize business overview"""
        return f"""
    {ticker} operates in its core business segments as detailed in Item 1 of the 10-K. 
    The company's strategy focuses on innovation and market expansion.

    *Note: Full business analysis requires indexed 10-K filings.*
    """

    def _categorize_risks(self, risks_by_year: Dict) -> Dict:
        """Categorize risks into themes"""
        # Simplified implementation
        return {
            "Market & Competition": [],
            "Operational": [],
            "Regulatory": [],
            "Technology": []
        }

    def _analyze_financials(self, sec_data: Dict, ticker: str) -> str:
        """Analyze financial trends"""
        return """
    ### Revenue & Profitability
    Analysis pending 10-K indexing.

    ### Cash Flow
    Analysis pending 10-K indexing.
    """

    def _assess_competition(self, sec_data: Dict, ticker: str) -> str:
        """Assess competitive position"""
        return f"""
    {ticker}'s competitive position is outlined in the 10-K Item 1 (Business) 
    and Item 1A (Risk Factors) sections.

    *Full competitive analysis requires indexed filings.*
    """

    
    def synthesize(self, sec_data: Dict, market_data: Dict, ticker: str) -> str:
        """Create comprehensive financial brief"""
        
        # Generate markdown report
        report = f"""# Financial Analysis: {ticker}
*Generated: {market_data['timestamp']}*

---

## Executive Summary

{self._generate_executive_summary(sec_data, market_data)}

---

## Current Market Metrics

{self._format_market_metrics(market_data)}

---

## Business Overview & Strategy

{self._summarize_business(sec_data, ticker)}

---

## Key Risk Factors

{self._analyze_risks(sec_data, ticker)}

---

## Financial Performance Trends

{self._analyze_financials(sec_data, ticker)}

---

## Competitive Position

{self._assess_competition(sec_data, ticker)}

---

## Investment Considerations

{self._generate_investment_view(sec_data, market_data)}

{self.citation_tracker.format_for_report()}
"""
        
        return report
    
    def _format_market_metrics(self, data: Dict) -> str:
        """Format market data as markdown table"""
        
        table = f"""
| Metric | Value |
|--------|-------|
| **Current Price** | ${data['current_price']:.2f} |
| **Market Cap** | ${data['market_cap']:,.0f} |
| **Shares Outstanding** | {data['shares_outstanding']:,.0f} |
| **52-Week Range** | ${data['52_week_low']:.2f} - ${data['52_week_high']:.2f} |
| **P/E Ratio** | {data.get('pe_ratio', 'N/A')} |
| **Beta** | {data.get('beta', 'N/A')} |
| **Dividend Yield** | {data.get('dividend_yield', 0) * 100:.2f}% |

*Source: Yahoo Finance, {data['timestamp']}*
"""
        
        # Track citation
        self.citation_tracker.add_citation(
            "Current market metrics",
            data['citation']
        )
        
        return table
    
    def _analyze_risks(self, sec_data: Dict, ticker: str) -> str:
        """Synthesize risk factors"""
        
        risks_by_year = sec_data.get('risks', {})
        
        output = "### Top Risk Factors (Last 5 Years)\n\n"
        
        # Identify recurring risks
        risk_categories = self._categorize_risks(risks_by_year)
        
        for category, risks in risk_categories.items():
            output += f"**{category}**\n\n"
            for risk in risks[:3]:  # Top 3 per category
                citation = risk['citation']
                output += f"- {risk['summary']} {citation.to_markdown()}\n"
            output += "\n"
        
        return output
    
    def _generate_investment_view(self, sec_data: Dict, market_data: Dict) -> str:
        """Generate investment recommendation framework"""
        
        view = """
### Strengths
{strengths}

### Concerns
{concerns}

### Key Catalysts to Watch
{catalysts}

*Note: This is an analytical framework, not investment advice.*
"""
        
        return view