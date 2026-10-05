from langgraph.graph import StateGraph, START, END
from app.agents.brain_agents.Neo4j_agent.state import BrainAgentState
from app.agents.brain_agents.Neo4j_agent.nodes.cypher_text_generation import (
    get_cypher_query,
    execute_cypher_node,
    final_answer,
)
from app.agents.brain_agents.Neo4j_agent.neo4j_main_node import agent_node
from app.agents.brain_agents.Neo4j_agent.neo4j_main_node import agent_node
MAX_ITERATIONS = 4


def route_from_agent(state: BrainAgentState) -> str:
    """Reads state, returns a node name. Never mutates anything."""
    if state["next_action"] == "write_final_answer":
        return "final_answer"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:  # hard cap — the safety net
        return "final_answer"
    return "generate_cypher"  # ← THE LOOP BACK


builder = StateGraph(BrainAgentState)
builder.add_node("agent", agent_node)
builder.add_node("generate_cypher", get_cypher_query)
builder.add_node("execute_cypher", execute_cypher_node)
builder.add_node("final_answer", final_answer)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_from_agent, ["generate_cypher", "final_answer"]
)
builder.add_edge("generate_cypher", "execute_cypher")
builder.add_edge("execute_cypher", "agent")  # ← THE EDGE THAT MAKES IT ReAct
builder.add_edge("final_answer", END)

neo4j_agent = builder.compile()
