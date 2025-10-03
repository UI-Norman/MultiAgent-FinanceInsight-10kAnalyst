import typer
from rich.console import Console
from rich.markdown import Markdown
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.vectordb.chroma import ChromaDb
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

app = typer.Typer()
console = Console()

# Simple in-memory session state
session_state = {
    "conversation": [],
    "retrieved_docs": {},
}

def initialize_agents(ticker: str):
    """Initialize all agents with local storage"""
    
    # Use ChromaDB (local file-based) instead of pgvector
    knowledge_base = AssistantKnowledge(
        vector_db=ChromaDb(
            collection=f"{ticker}_10k",
            path="./chroma_db"  # Local storage
        ),
    )
    
    # SEC Researcher Agent
    sec_agent = Agent(
        name="SEC Researcher",
        role="Analyze 10-K filings",
        model=OpenAIChat(id="gpt-4o"),
        knowledge_base=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Extract key information from 10-K filings",
            "Compare trends across years",
            "Always cite: [Ticker 10-K Year, Section]",
            "Focus on risks, financial performance, and strategy"
        ],
        show_tool_calls=True,
        markdown=True,
    )
    
    # Market Data Agent
    market_agent = Agent(
        name="Market Data Agent",
        role="Fetch real-time market data",
        model=OpenAIChat(id="gpt-4o"),
        tools=[YFinanceTools(
            stock_price=True,
            company_info=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
        )],
        instructions=[
            "Get current price, market cap, shares outstanding",
            "Include 52-week range and P/E ratio",
            "Always include timestamp",
            "Format data in tables"
        ],
        show_tool_calls=True,
        markdown=True,
    )
    
    # Financial Analyst (Synthesizer)
    analyst_agent = Agent(
        name="Financial Analyst",
        role="Synthesize comprehensive financial analysis",
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            "Create a concise markdown brief with:",
            "1. Executive Summary",
            "2. Current Market Metrics (use tables)",
            "3. Key Risk Factors",
            "4. Financial Performance Trends",
            "5. Investment Considerations",
            "Cite all claims from 10-Ks or market data",
            "Keep it under 1000 words"
        ],
        markdown=True,
    )
    
    # Orchestrator Agent Team
    orchestrator = Agent(
        name="Orchestrator",
        team=[sec_agent, market_agent, analyst_agent],
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            "Coordinate the analysis workflow:",
            "1. First, get market data from Market Data Agent",
            "2. Then, get 10-K insights from SEC Researcher",
            "3. Finally, ask Financial Analyst to synthesize everything",
            "Ensure all data is cited properly"
        ],
        show_tool_calls=True,
        markdown=True,
    )
    
    return orchestrator, sec_agent, market_agent, analyst_agent


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., NVDA, AAPL)"),
    simple: bool = typer.Option(False, help="Simple mode (market data only)"),
):
    """
    Analyze a company using multi-agent system
    """
    
    console.print(f"\n[bold blue]🔍 Analyzing {ticker}...[/bold blue]\n")
    
    try:
        if simple:
            # Simple mode - just market data
            console.print("[yellow]📊 Fetching market data only...[/yellow]")
            
            market_agent = Agent(
                name="Market Agent",
                model=OpenAIChat(id="gpt-4o"),
                tools=[YFinanceTools(
                    stock_price=True,
                    company_info=True,
                    analyst_recommendations=True,
                )],
                markdown=True,
            )
            
            response = market_agent.run(
                f"Get comprehensive market data for {ticker} including current price, "
                f"market cap, P/E ratio, 52-week range. Format as a table."
            )
            
            console.print("\n" + "="*80 + "\n")
            console.print(Markdown(response.content))
            console.print("\n" + "="*80 + "\n")
            
        else:
            # Full multi-agent analysis
            console.print("[yellow]⚙️  Initializing multi-agent system...[/yellow]")
            orchestrator, sec_agent, market_agent, analyst_agent = initialize_agents(ticker)
            
            console.print("[yellow]📋 Creating analysis plan...[/yellow]")
            
            # Run orchestrated analysis
            query = f"""
            Create a comprehensive financial analysis for {ticker}:
            
            1. Get current market metrics (price, market cap, P/E, etc.)
            2. Analyze 10-K filings for business risks and financial trends
            3. Synthesize into a brief markdown report with:
               - Executive Summary
               - Current Market Metrics
               - Key Risk Factors
               - Investment Considerations
            
            Cite all sources appropriately.
            """
            
            console.print("[yellow]🤖 Running multi-agent workflow...[/yellow]\n")
            
            response = orchestrator.run(query)
            
            # Display report
            console.print("\n" + "="*80 + "\n")
            console.print(Markdown(response.content))
            console.print("\n" + "="*80 + "\n")
            
            # Save to session state
            session_state["conversation"].append({
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "query": query,
                "response": response.content
            })
        
        console.print("[bold green]✅ Analysis complete![/bold green]\n")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


@app.command()
def chat(
    ticker: str = typer.Argument(..., help="Stock ticker"),
):
    """
    Interactive chat mode for follow-up questions
    """
    
    console.print(f"\n[bold blue]💬 Chat mode for {ticker}[/bold blue]")
    console.print("[dim]Type 'exit' to quit, 'history' to see conversation[/dim]\n")
    
    # Initialize simple agent for chat
    chat_agent = Agent(
        name="Financial Chat Assistant",
        model=OpenAIChat(id="gpt-4o"),
        tools=[YFinanceTools(
            stock_price=True,
            company_info=True,
            analyst_recommendations=True,
        )],
        instructions=[
            f"You are analyzing {ticker}",
            "Answer questions about the company's financials and market data",
            "Use the YFinance tool to get current data",
            "Keep responses concise but informative"
        ],
        markdown=True,
    )
    
    while True:
        try:
            query = typer.prompt("\n[You]")
            
            if query.lower() == 'exit':
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            
            if query.lower() == 'history':
                console.print("\n[bold]Conversation History:[/bold]")
                for turn in session_state["conversation"]:
                    console.print(f"\n[dim]{turn['timestamp']}[/dim]")
                    console.print(f"Query: {turn['query'][:100]}...")
                continue
            
            # Get response
            response = chat_agent.run(query)
            
            console.print("\n[Assistant]")
            console.print(Markdown(response.content))
            
            # Save to session
            session_state["conversation"].append({
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "query": query,
                "response": response.content
            })
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def info():
    """Show system information"""
    
    console.print("\n[bold blue]ℹ️  System Information[/bold blue]\n")
    
    info_text = """
    ## Multi-Agent Financial Analysis System
    
    **Storage:**
    - Vector DB: ChromaDB (local files in ./chroma_db)
    - Session: In-memory Python dict
    
    **Agents:**
    - SEC Researcher: Analyzes 10-K filings
    - Market Data Agent: Real-time data from Yahoo Finance
    - Financial Analyst: Synthesizes reports
    - Orchestrator: Coordinates workflow
    
    **Usage:**
    ```bash
    # Simple analysis (market data only)
    python main.py analyze NVDA --simple
    
    # Full analysis (requires 10-K data indexed)
    python main.py analyze NVDA
    
    # Interactive chat
    python main.py chat NVDA
    ```
    
    **Note:** This is the simplified no-Docker version.
    For production features, use the full Docker setup.
    """
    
    console.print(Markdown(info_text))


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]❌ OPENAI_API_KEY not found in environment![/bold red]")
        console.print("Please set it in .env file or export it:")
        console.print("  export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    app()