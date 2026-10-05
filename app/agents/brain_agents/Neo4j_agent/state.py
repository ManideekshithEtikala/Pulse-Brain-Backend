from __future__ import annotations

from typing import TypedDict, Annotated , Optional
from langchain_core.messages import BaseMessage



class BrainAgentState(TypedDict):
    """
    A dictionary that represents the state of a Brain Agent.
    """
    user_query:str
    cypher_query : str
    error: Optional[str]
    node_results : Optional[list[dict]]
    iteration_count: int
    next_action:str
    final_result:str
