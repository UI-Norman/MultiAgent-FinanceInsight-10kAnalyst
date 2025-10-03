from phi.agent import Agent
from typing import List, Dict
import json
from phi.model.openai import OpenAIChat


class OrchestratorAgent:
    """
    Plans and coordinates the multi-agent workflow
    """
    
    def __init__(self, sec_agent, market_agent, analyst_agent, auditor_agent):
        self.sec_agent = sec_agent
        self.market_agent = market_agent
        self.analyst_agent = analyst_agent
        self.auditor_agent = auditor_agent
        # Fix the model
        self.planner = Agent(
            name="Planner",
            model=OpenAIChat(id="gpt-4o"),  # ADD THIS LINE
            role="Decompose queries and create execution plan",
            instructions=[
                "Break complex queries into sequential steps",
                "Identify which agents are needed",
                "Create a DAG of task dependencies"
            ]
        )

        # self.planner = Agent(
        #     name="Planner",
        #     role="Decompose queries and create execution plan",
        #     instructions=[
        #         "Break complex queries into sequential steps",
        #         "Identify which agents are needed",
        #         "Create a DAG of task dependencies"
        #     ]
        # )
    
    def create_plan(self, user_query: str) -> Dict:
        """Generate execution plan"""
        
        plan_prompt = f"""
        User Query: {user_query}
        
        Create an execution plan:
        1. What information is needed from 10-K filings?
        2. What market data is required?
        3. What comparisons or calculations are needed?
        4. In what order should agents execute?
        
        Return as JSON:
        {{
            "steps": [
                {{"agent": "sec_researcher", "task": "...", "dependencies": []}},
                {{"agent": "market_data", "task": "...", "dependencies": []}},
                {{"agent": "analyst", "task": "...", "dependencies": ["sec_researcher", "market_data"]}},
                {{"agent": "auditor", "task": "...", "dependencies": ["analyst"]}}
            ]
        }}
        """
        
        response = self.planner.run(plan_prompt)
        plan = json.loads(response.content)
        
        return plan
    
    def execute_plan(self, plan: Dict, ticker: str) -> Dict:
        """Execute the plan by calling agents in order"""
        
        results = {}
        
        for step in plan['steps']:
            agent_name = step['agent']
            task = step['task']
            
            # Wait for dependencies
            deps_ready = all(dep in results for dep in step['dependencies'])
            
            if not deps_ready:
                continue
            
            # Execute agent
            if agent_name == "sec_researcher":
                results[agent_name] = self.sec_agent.run(task, ticker)
            elif agent_name == "market_data":
                results[agent_name] = self.market_agent.run(ticker)
            elif agent_name == "analyst":
                results[agent_name] = self.analyst_agent.synthesize(
                    results.get('sec_researcher'),
                    results.get('market_data')
                )
            elif agent_name == "auditor":
                results[agent_name] = self.auditor_agent.verify(results['analyst'])
        
        return results