import typer
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
import os
from agents.orchestrator import OrchestratorAgent
from agents.sec_researcher import SECResearcherAgent
from agents.market_data import MarketDataAgent
from agents.analyst import FinancialAnalystAgent
from agents.auditor import AuditorAgent
from core.memory import GlobalMemory, SessionMemory
from core.knowledge_base import create_knowledge_base
from core.rag_pipeline import AdvancedRAGPipeline
from utils.logger import StructuredLogger
from utils.cost_tracker import CostTracker
load_dotenv()

app = typer.Typer()
console = Console()
logger = StructuredLogger("main")

@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g., NVDA, AAPL)"),
    user_id: str = typer.Option("default_user", help="User ID for personalization"),
    new_session: bool = typer.Option(False, help="Start new session"),
    simple: bool = typer.Option(False, help="Simple mode (market data only)")
):
    """
    Analyze a company using multi-agent system
    """
    
    console.print(f"[bold blue]🔍 Analyzing {ticker}...[/bold blue]\n")
    
    # Initialize memory
    global_memory = GlobalMemory()
    session_memory = SessionMemory()
    cost_tracker = CostTracker()
    
    try:
        if simple:
            # Simple mode - market data only
            console.print("[yellow]📊 Running simple analysis (market data only)...[/yellow]")
            
            market_agent = MarketDataAgent()
            market_data = market_agent.run(ticker)
            
            # Simple report
            report = f"""# Market Analysis: {ticker}

## Current Market Metrics

| Metric | Value |
|--------|-------|
| **Current Price** | ${market_data.get('current_price', 0):.2f} |
| **Market Cap** | ${market_data.get('market_cap', 0):,.0f} |
| **Shares Outstanding** | {market_data.get('shares_outstanding', 0):,.0f} |
| **52-Week Range** | ${market_data.get('52_week_low', 0):.2f} - ${market_data.get('52_week_high', 0):.2f} |
| **P/E Ratio** | {market_data.get('pe_ratio', 'N/A')} |
| **Beta** | {market_data.get('beta', 'N/A')} |

*Source: Yahoo Finance, {market_data.get('timestamp')}*
"""
            
            console.print("\n" + "="*80 + "\n")
            console.print(Markdown(report))
            console.print("\n" + "="*80 + "\n")
            
        else:
            # Full multi-agent analysis
            console.print("[yellow]⚙️  Initializing multi-agent system...[/yellow]")
            
            # Initialize knowledge base and RAG
            knowledge_base = create_knowledge_base(ticker)
            rag_pipeline = AdvancedRAGPipeline()
            
            # Initialize agents
            sec_agent = SECResearcherAgent(knowledge_base, rag_pipeline)
            market_agent = MarketDataAgent()
            analyst_agent = FinancialAnalystAgent()
            auditor_agent = AuditorAgent()
            
            orchestrator = OrchestratorAgent(
                sec_agent, market_agent, analyst_agent, auditor_agent
            )
            
            # Execute workflow
            # Step 1: Create plan
            console.print("[yellow]📋 Creating execution plan...[/yellow]")
            plan = orchestrator.create_plan(f"Analyze {ticker}")
            logger.log_agent_call("orchestrator", "planning", 0.5)
            
            # Step 2: Execute plan
            console.print("[yellow]⚙️  Executing multi-agent workflow...[/yellow]")
            results = orchestrator.execute_plan(plan, ticker)
            
            # Step 3: Get final report
            report = results.get('analyst', 'No report generated')
            
            # Step 4: Audit
            console.print("[yellow]✅ Auditing report...[/yellow]")
            audit_results = results.get('auditor', {})
            
            # Display report
            console.print("\n" + "="*80 + "\n")
            console.print(Markdown(report))
            console.print("\n" + "="*80 + "\n")
            
            # Display audit results
            if audit_results:
                confidence = audit_results.get('overall_confidence', 0)
                console.print(f"[bold]Audit Confidence:[/bold] {confidence:.1%}")
                
                citation_check = audit_results.get('citation_check', {})
                if citation_check.get('uncited_claims'):
                    console.print("[yellow]⚠️  Some claims lack citations[/yellow]")
            
            # Save to global memory
            global_memory.save_analysis(
                ticker=ticker,
                summary=str(report)[:500],
                metrics={},
                embedding=[]
            )
        
        # Print cost summary
        cost_tracker.print_summary()
        
        console.print("[bold green]✅ Analysis complete![/bold green]\n")
        
    except Exception as e:
        logger.logger.error(f"Analysis failed: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        raise

@app.command()
def chat(
    ticker: str = typer.Argument(..., help="Stock ticker"),
    user_id: str = typer.Option("default_user")
):
    """
    Interactive chat mode for follow-up questions
    """
    
    console.print(f"[bold blue]💬 Chat mode for {ticker}[/bold blue]")
    console.print("[dim]Type 'exit' to quit[/dim]\n")
    
    session_memory = SessionMemory()
    market_agent = MarketDataAgent()
    
    while True:
        try:
            query = typer.prompt("You")
            
            if query.lower() == 'exit':
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            
            # Add to session memory
            session_memory.add_turn("user", query)
            
            # Simple response using market agent
            if "price" in query.lower() or "market" in query.lower():
                data = market_agent.run(ticker)
                response = f"Current price: ${data.get('current_price', 0):.2f}, Market cap: ${data.get('market_cap', 0):,.0f}"
            else:
                response = f"Question about {ticker}: {query}\n\n(Full chat requires additional implementation)"
            
            console.print(f"\n[Assistant] {response}\n")
            session_memory.add_turn("assistant", response)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

@app.command()
def info():
    """Show system information"""
    
    info_text = """
# Multi-Agent Financial Analysis System

## Status
- ✅ Core infrastructure ready
- ✅ Market data integration (Yahoo Finance)
- ⚠️  10-K RAG (requires indexing)
- ✅ Multi-agent coordination

## Quick Start
```bash
# Simple analysis (works immediately)
python main.py analyze NVDA --simple

# Full analysis (requires 10-K data)
python main.py analyze NVDA

# Interactive chat
python main.py chat NVDA
```

## Next Steps
1. Index 10-K filings for full analysis
2. Enhance RAG pipeline
3. Add comprehensive tests
"""
    
    console.print(Markdown(info_text))

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]❌ OPENAI_API_KEY not found![/bold red]")
        console.print("Set it in .env file or:")
        console.print("  export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    app()