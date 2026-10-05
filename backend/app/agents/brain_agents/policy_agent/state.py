from typing import TypedDict, List, Dict, Any, Optional


class PolicyAgentState(TypedDict, total=False):

    user_query: str

    # Trusted application context
    user_context: Dict[str, Any]

    # --- query_analysis_node outputs ---
    intent: str
    policy_topic: Optional[str]
    enhanced_query: str
    search_queries: List[str]
    metadata_filters: Dict[str, Any]

    # --- hybrid_retrieval_node output (overwritten by rerank_node) ---
    retrieval_results: List[Dict[str, Any]]

    # --- rerank_node output ---
    low_confidence: Optional[bool]  # True = no chunk cleared the score floor

    # --- future generation node outputs (declare now, fill later) ---
    # answer: Optional[str]
    # citations: List[Dict[str, Any]]
    # --- policy_answer_node outputs ---
    answer: Optional[str]
    citations: List[Dict[str, Any]]
    answer_confidence: Optional[
        str
    ]  # fully_answered | partially_answered | not_found_in_evidence
    # --- answer_validator_node output ---
    validation: Optional[Dict[str, Any]]  # full verdict object (see validator node)

    # --- refinement loop (node 8) ---
    refinement_attempts: int  # how many refine cycles have run
    refinement_log: List[Dict[str, Any]]  # audit trail of each refinement
    error: Optional[str]
