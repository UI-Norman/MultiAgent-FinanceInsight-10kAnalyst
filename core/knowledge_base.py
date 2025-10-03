from phi.knowledge import AssistantKnowledge
from phi.vectordb.chroma import ChromaDb

def create_knowledge_base(ticker: str):
    """Create knowledge base for a ticker"""
    knowledge_base = AssistantKnowledge(
        vector_db=ChromaDb(
            collection=f"{ticker}_10k",
            path="./chroma_db"
        ),
    )
    return knowledge_base

