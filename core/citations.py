from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Citation:
    source_type: str  # "10-K" or "market_data"
    ticker: str
    year: Optional[str]
    section: Optional[str]
    page_range: Optional[Tuple[int, int]] = None
    url: str = ""
    timestamp: Optional[str] = None
    
    def to_markdown(self) -> str:
        if self.source_type == "10-K":
            return f"[{self.ticker} 10-K {self.year}, {self.section}]({self.url})"
        else:
            return f"[Market Data: {self.ticker} at {self.timestamp}]"

class CitationTracker:
    """Track and validate citations throughout analysis"""
    
    def __init__(self):
        self.citations: List[Citation] = []
        self.claim_to_citation: Dict[str, List[Citation]] = {}
    
    def add_citation(self, claim: str, citation: Citation):
        """Link a claim to its source"""
        if claim not in self.claim_to_citation:
            self.claim_to_citation[claim] = []
        
        self.claim_to_citation[claim].append(citation)
        self.citations.append(citation)
    
    def validate_citations(self) -> Dict[str, bool]:
        """Check that all claims have valid citations"""
        validation_results = {}
        
        for claim, citations in self.claim_to_citation.items():
            validation_results[claim] = len(citations) > 0
        
        return validation_results
    
    def format_for_report(self) -> str:
        """Generate citations section for markdown report"""
        if not self.citations:
            return ""
        
        output = "\n\n## Sources\n\n"
        
        seen = set()
        for i, citation in enumerate(self.citations, 1):
            citation_str = citation.to_markdown()
            if citation_str not in seen:
                output += f"{i}. {citation_str}\n"
                seen.add(citation_str)
        
        return output