"""
Shared application state.
Holds the global QA chain instance to avoid circular imports.
"""

from src.llm.qa_chain import QAChain

qa_chain: QAChain | None = None