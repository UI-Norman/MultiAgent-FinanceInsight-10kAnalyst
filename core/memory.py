import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import chromadb

class GlobalMemory:
    """Persistent memory across sessions"""
    
    def __init__(self, db_path: str = "global_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.vector_db = chromadb.Client()
        self.collection = self.vector_db.create_collection("company_summaries")
        self._init_tables()
    
    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                risk_taxonomy JSON,
                writing_style TEXT,
                preferred_kpis JSON,
                version INTEGER,
                created_at TIMESTAMP,
                ttl_expires TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                analysis_date TIMESTAMP,
                summary TEXT,
                key_metrics JSON,
                embedding BLOB
            )
        """)
        
        self.conn.commit()
    
    def save_user_preferences(self, user_id: str, preferences: Dict):
        """Save user preferences with versioning and TTL"""
        ttl = datetime.now() + timedelta(days=365)  # 1 year TTL
        
        self.conn.execute("""
            INSERT OR REPLACE INTO user_preferences 
            (user_id, risk_taxonomy, writing_style, preferred_kpis, version, created_at, ttl_expires)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            json.dumps(preferences.get('risk_taxonomy', {})),
            preferences.get('writing_style', 'narrative'),
            json.dumps(preferences.get('kpis', [])),
            preferences.get('version', 1),
            datetime.now(),
            ttl
        ))
        self.conn.commit()
    
    def get_similar_companies(self, ticker: str, n: int = 3) -> List[Dict]:
        """Find similar previously analyzed companies"""
        results = self.collection.query(
            query_texts=[ticker],
            n_results=n
        )
        return results
    
    def save_analysis(self, ticker: str, summary: str, metrics: Dict, embedding: List[float]):
        """Save company analysis with vector embedding"""
        self.conn.execute("""
            INSERT INTO analysis_history (ticker, analysis_date, summary, key_metrics, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (ticker, datetime.now(), summary, json.dumps(metrics), json.dumps(embedding)))
        
        # Also save to vector DB for similarity search
        self.collection.add(
            documents=[summary],
            metadatas=[{"ticker": ticker, "date": str(datetime.now())}],
            ids=[f"{ticker}_{datetime.now().timestamp()}"]
        )
        
        self.conn.commit()

class SessionMemory:
    """Ephemeral memory for current conversation"""
    
    def __init__(self):
        self.conversation_history = []
        self.intermediate_plans = []
        self.retrieved_documents = {}
        self.agent_states = {}
    
    def add_turn(self, role: str, message: str, metadata: Dict = None):
        """Add conversation turn"""
        self.conversation_history.append({
            "role": role,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })
    
    def save_plan(self, plan: Dict):
        """Save orchestrator's plan"""
        self.intermediate_plans.append(plan)
    
    def cache_retrieval(self, query: str, results: List):
        """Cache retrieved documents to avoid re-fetching"""
        self.retrieved_documents[query] = results
    
    def get_context_window(self, n_turns: int = 5) -> List[Dict]:
        """Get last N conversation turns"""
        return self.conversation_history[-n_turns:]