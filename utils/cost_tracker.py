from typing import Dict
from dataclasses import dataclass, field

@dataclass
class CostTracker:
    """Track API and LLM costs"""
    
    llm_costs: Dict[str, float] = field(default_factory=dict)
    api_costs: Dict[str, float] = field(default_factory=dict)
    
    # Pricing (update with actual rates)
    PRICING = {
        "gpt-4o-input": 0.005 / 1000,  # per token
        "gpt-4o-output": 0.015 / 1000,
        "embedding-ada-002": 0.0001 / 1000,
        "yfinance": 0.0  # Free
    }
    
    def track_llm_call(self, model: str, input_tokens: int, output_tokens: int):
        """Track LLM API call"""
        
        cost = (
            input_tokens * self.PRICING.get(f"{model}-input", 0) +
            output_tokens * self.PRICING.get(f"{model}-output", 0)
        )
        
        if model not in self.llm_costs:
            self.llm_costs[model] = 0
        
        self.llm_costs[model] += cost
    
    def track_api_call(self, api_name: str, cost: float = 0.0):
        """Track external API call"""
        
        if api_name not in self.api_costs:
            self.api_costs[api_name] = 0
        
        self.api_costs[api_name] += cost
    
    def get_summary(self) -> Dict:
        """Get cost summary"""
        
        total_llm = sum(self.llm_costs.values())
        total_api = sum(self.api_costs.values())
        
        return {
            "total_cost_usd": total_llm + total_api,
            "llm_cost_breakdown": self.llm_costs,
            "api_cost_breakdown": self.api_costs,
            "dominant_cost": "LLM" if total_llm > total_api else "API"
        }
    
    def print_summary(self):
        """Print formatted cost summary"""
        
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("💰 COST SUMMARY")
        print("="*50)
        print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f"\nLLM Costs:")
        for model, cost in summary['llm_cost_breakdown'].items():
            print(f"  - {model}: ${cost:.4f}")
        print(f"\nAPI Costs:")
        for api, cost in summary['api_cost_breakdown'].items():
            print(f"  - {api}: ${cost:.4f}")
        print("="*50 + "\n")