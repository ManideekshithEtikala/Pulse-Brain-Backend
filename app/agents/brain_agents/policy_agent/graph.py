"""
Policy Agent Graph (full pipeline with validation loop)

  START -> query_analysis -> [err?] -> hybrid_retrieval -> [err?] -> rerank
        -> [err?] -> policy_answer -> [err?] -> answer_validator
        -> PASS                                   -> END
        -> FAIL + refinement helps + attempts left -> retrieval_refinement
        -> FAIL otherwise                          -> END (final gate already
                                                      shipped a safe answer)

  retrieval_refinement -> hybrid_retrieval  (cycle; bounded by
                             refinement_attempts <= MAX_REFINEMENT_ATTEMPTS)

Every FAIL exit path still ends with a shippable answer in state — the
validator's final gate guarantees it.
"""

import json
import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from app.agents.brain_agents.policy_agent.state import PolicyAgentState
from app.agents.brain_agents.policy_agent.nodes.query_analysis_node import (
    query_enhancer_node,
)
from app.agents.brain_agents.policy_agent.nodes.hybrid_retreival import (
    hybrid_retrieval_node,
)
from app.agents.brain_agents.policy_agent.nodes.rerank_node import rerank_node
from app.agents.brain_agents.policy_agent.nodes.policy_agent import policy_answer_node
from app.agents.brain_agents.policy_agent.nodes.answer_validator import (
    answer_validator_node,
    MAX_REFINEMENT_ATTEMPTS,
)
from app.agents.brain_agents.policy_agent.nodes.refinement_node import (
    retrieval_refinement_node,
)

logger = logging.getLogger(__name__)
VERBOSE = False


def route_on_error(state: PolicyAgentState) -> str:
    if state.get("error"):
        return "end"
    return "continue"


def route_after_validation(state: PolicyAgentState) -> str:
    """PASS -> done. FAIL -> refine only if it would help AND budget remains."""
    if state.get("error"):
        return "end"
    validation = state.get("validation") or {}
    if validation.get("overall") == "pass":
        return "pass"
    if state.get("refinement_attempts", 0) >= MAX_REFINEMENT_ATTEMPTS:
        return "end"
    if not validation.get("refinement_would_help", False):
        return "end"
    return "refine"


def build_policy_agent_graph():
    graph = StateGraph(PolicyAgentState)

    graph.add_node("query_analysis", query_enhancer_node)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("policy_answer", policy_answer_node)
    graph.add_node("answer_validator", answer_validator_node)
    graph.add_node("retrieval_refinement", retrieval_refinement_node)

    graph.add_edge(START, "query_analysis")

    graph.add_conditional_edges(
        "query_analysis",
        route_on_error,
        {"continue": "hybrid_retrieval", "end": END},
    )
    graph.add_conditional_edges(
        "hybrid_retrieval",
        route_on_error,
        {"continue": "rerank", "end": END},
    )
    graph.add_conditional_edges(
        "rerank",
        route_on_error,
        {"continue": "policy_answer", "end": END},
    )
    graph.add_conditional_edges(
        "policy_answer",
        route_on_error,
        {"continue": "answer_validator", "end": END},
    )
    graph.add_conditional_edges(
        "answer_validator",
        route_after_validation,
        {"pass": END, "refine": "retrieval_refinement", "end": END},
    )

    # THE CYCLE: refinement feeds straight back into retrieval. The loop is
    # bounded by refinement_attempts in route_after_validation.
    graph.add_edge("retrieval_refinement", "hybrid_retrieval")

    return graph.compile()


policy_agent_graph = build_policy_agent_graph()


# # ---------------------------------------------------------
# # Printing helpers
# # ---------------------------------------------------------


# def print_node_update(node_name: str, update: Dict[str, Any]) -> None:
#     print(f"\n--- state update after node: {node_name} ---")
#     if VERBOSE:
#         print(json.dumps(update, indent=2, default=str))
#         return
#     summary: Dict[str, Any] = {}
#     for key, value in update.items():
#         if key == "retrieval_results":
#             summary[key] = f"[{len(value)} chunks]"
#         elif key == "refinement_log":
#             summary[key] = [e.get("strategy_note") for e in value]
#         else:
#             summary[key] = value
#     print(json.dumps(summary, indent=2, default=str))


# def print_final_results(final_state: Dict[str, Any]) -> None:
#     print("\n" + "=" * 100)
#     print("FINAL STATE (all nodes merged)")
#     print("=" * 100)
#     print(f"intent            : {final_state.get('intent')}")
#     print(f"enhanced_query    : {final_state.get('enhanced_query')}")
#     print(f"metadata_filters  : {final_state.get('metadata_filters')}")
#     print(f"refinement_attempts: {final_state.get('refinement_attempts', 0)}")
#     print(f"low_confidence    : {final_state.get('low_confidence')}")
#     v = final_state.get("validation") or {}
#     if v:
#         print(f"validation        : overall={v.get('overall')}")
#         for issue in v.get("issues", []) or []:
#             print(f"    - {issue}")
#     results = final_state.get("retrieval_results") or []
#     print(f"\nEVIDENCE SHIPPED: {len(results)} chunks")
#     for i, chunk in enumerate(results, start=1):
#         md = chunk.get("metadata", {})
#         print(
#             f"  {i}. rerank={chunk.get('rerank_score', 0):.3f} "
#             f"§{md.get('section_number', '?')} {md.get('section_title', '')}"
#         )
#     print("\nANSWER " + "-" * 92)
#     print(final_state.get("answer"))
#     print("\nANSWER CONFIDENCE:", final_state.get("answer_confidence"))
#     for c in final_state.get("citations") or []:
#         print(
#             f"  {c['marker']} {c['document_title']} (v{c['version']}) — "
#             f"§{c['section_number']} {c['section_title']}"
#         )
#     print("=" * 100)


# if __name__ == "__main__":
#     test_queries = [
#         "I need a leave on 25th of September 2025",
#         "What does the Attendance and Leave Policy say about the grace period?",
#     ]

#     for q in test_queries:
#         print("\n" + "#" * 100)
#         print("USER QUERY:", q)
#         print("#" * 100)

#         initial_state: PolicyAgentState = {"user_query": q}
#         running_state: Dict[str, Any] = dict(initial_state)

#         for step in policy_agent_graph.stream(initial_state, stream_mode="updates"):
#             for node_name, update in step.items():
#                 print_node_update(node_name, update)
#                 running_state.update(update)

#         print_final_results(running_state)
