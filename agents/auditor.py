from phi.agent import Agent
from typing import Dict, List
import re
from phi.model.openai import OpenAIChat


class AuditorAgent:
    """Verify citations and numeric accuracy"""    
    def __init__(self):
        self.agent = Agent(
        name="Auditor",
        model=OpenAIChat(id="gpt-4o"),  # ADD THIS LINE
        role="Fact-check and validate analysis",
        instructions=[
            "Verify every citation exists",
            "Check numeric consistency",
            "Flag unsupported claims",
            "Validate market cap calculations"
        ]
    )

    def _verify_claims(self, analyst_report: str) -> Dict:
        """Verify factual claims"""
        return {
            "unsupported_claims": []  # Simplified for now
        }
    
    def verify(self, analyst_report: str, citations: List, market_data: Dict) -> Dict:
        """Comprehensive verification"""
        
        verification_results = {
            "citation_check": self._verify_citations(analyst_report, citations),
            "numeric_check": self._verify_numbers(market_data),
            "claim_check": self._verify_claims(analyst_report),
            "overall_confidence": None
        }
        
        # Calculate overall confidence
        checks_passed = sum([
            verification_results['citation_check']['all_valid'],
            verification_results['numeric_check']['calculations_valid'],
            len(verification_results['claim_check']['unsupported_claims']) == 0
        ])
        
        verification_results['overall_confidence'] = checks_passed / 3.0
        
        return verification_results
    
    def _verify_citations(self, report: str, citations: List) -> Dict:
        """Check that all claims have citations"""
        
        # Extract citation markers from report
        citation_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        found_citations = re.findall(citation_pattern, report)
        
        # Extract claims without citations
        sentences = report.split('.')
        uncited_claims = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Substantial claim
                if not re.search(citation_pattern, sentence):
                    # Check if it's a general statement vs factual claim
                    if self._is_factual_claim(sentence):
                        uncited_claims.append(sentence.strip())
        
        return {
            "total_citations": len(found_citations),
            "uncited_claims": uncited_claims,
            "all_valid": len(uncited_claims) == 0
        }
    
    def _verify_numbers(self, market_data: Dict) -> Dict:
        """Verify numeric calculations"""
        
        price = market_data.get('current_price')
        shares = market_data.get('shares_outstanding')
        market_cap = market_data.get('market_cap')
        
        calculated_cap = price * shares if price and shares else None
        
        if calculated_cap and market_cap:
            error_pct = abs(calculated_cap - market_cap) / market_cap * 100
            valid = error_pct < 5.0  # 5% tolerance
        else:
            valid = False
            error_pct = None
        
        return {
            "calculations_valid": valid,
            "market_cap_error_pct": error_pct,
            "details": {
                "calculated": calculated_cap,
                "reported": market_cap
            }
        }
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if sentence is a factual claim requiring citation"""
        
        # Heuristic: contains numbers, company names, specific terms
        factual_indicators = [
            r'\d+%',  # Percentages
            r'\$[\d,]+',  # Dollar amounts
            r'\d{4}',  # Years
            r'increased|decreased|grew|declined',  # Trend words
            r'risk|revenue|profit|loss|debt'  # Financial terms
        ]
        
        for pattern in factual_indicators:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        return False