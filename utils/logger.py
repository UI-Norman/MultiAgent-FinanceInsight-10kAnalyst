# Utils.logger
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Structured JSON logging for observability"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self._json_formatter())
        self.logger.addHandler(handler)
    
    def _json_formatter(self):
        """Custom JSON formatter"""
        
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                
                # Add extra fields
                if hasattr(record, 'extra'):
                    log_data.update(record.extra)
                
                return json.dumps(log_data)
        
        return JsonFormatter()
    
    def log_retrieval(self, query: str, results: List, strategy: str):
        """Log retrieval operation"""
        self.logger.info(
            "Retrieval completed",
            extra={
                "event_type": "retrieval",
                "query": query,
                "num_results": len(results),
                "strategy": strategy,
                "top_scores": [r.score for r in results[:3]]
            }
        )
    
    def log_agent_call(self, agent_name: str, task: str, duration: float):
        """Log agent execution"""
        self.logger.info(
            "Agent completed task",
            extra={
                "event_type": "agent_execution",
                "agent": agent_name,
                "task": task,
                "duration_seconds": duration
            }
        )
    
    def log_api_call(self, api: str, endpoint: str, status: int, cost: float = None):
        """Log external API calls"""
        log_data = {
            "event_type": "api_call",
            "api": api,
            "endpoint": endpoint,
            "status": status
        }
        
        if cost:
            log_data["estimated_cost_usd"] = cost
        
        self.logger.info("API call", extra=log_data)