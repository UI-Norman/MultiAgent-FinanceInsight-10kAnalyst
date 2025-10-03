from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

@dataclass
class RetrievalResult:
    content: str
    metadata: Dict
    score: float
    source: str  # "dense", "sparse", "hybrid"

class AdvancedRAGPipeline:
    """
    Multi-stage retrieval:
    1. Query decomposition
    2. Hybrid search (dense + sparse)
    3. Re-ranking with cross-encoder
    4. Cross-document comparison
    """
    
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents
        
        # Sparse retrieval (BM25)
        tokenized_docs = [doc.content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Re-ranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def decompose_query(self, query: str) -> List[str]:
        """Break complex query into sub-queries"""
        # Use LLM to decompose
        prompt = f"""Break this query into 2-3 focused sub-queries:
        Query: {query}
        
        Return as JSON list: ["sub-query 1", "sub-query 2"]
        """
        # Call LLM...
        return sub_queries
    
    def hybrid_search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Combine dense + sparse retrieval"""
        
        # Dense retrieval (vector similarity)
        dense_results = self.vector_store.similarity_search(query, k=k)
        
        # Sparse retrieval (BM25)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = np.argsort(bm25_scores)[-k:][::-1]
        sparse_results = [self.documents[i] for i in sparse_indices]
        
        # Merge and deduplicate
        combined = self._merge_results(dense_results, sparse_results)
        return combined
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank using cross-encoder"""
        pairs = [[query, r.content] for r in results]
        scores = self.reranker.predict(pairs)
        
        for result, score in zip(results, scores):
            result.score = score
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def compare_across_years(self, query: str, years: List[str]) -> Dict[str, List[RetrievalResult]]:
        """Retrieve same topic across multiple years"""
        results_by_year = {}
        
        for year in years:
            # Add year filter to metadata
            filtered_query = f"{query} (year: {year})"
            results = self.hybrid_search(filtered_query, k=5)
            results_by_year[year] = results
        
        return results_by_year
    
    def retrieve(self, query: str, strategy: str = "hybrid") -> List[RetrievalResult]:
        """Main retrieval entry point"""
        
        # Step 1: Decompose if complex
        sub_queries = self.decompose_query(query)
        
        all_results = []
        for sq in sub_queries:
            # Step 2: Hybrid search
            results = self.hybrid_search(sq, k=20)
            
            # Step 3: Re-rank
            results = self.rerank(sq, results)
            
            all_results.extend(results[:5])  # Top 5 per sub-query
        
        # Step 4: Final deduplication
        unique_results = self._deduplicate(all_results)
        
        return unique_results[:10]  # Return top 10 overall