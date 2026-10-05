# agent_brain.py
from pydantic import BaseModel
from typing import Literal
from app.agents.brain_agents.Neo4j_agent.state import BrainAgentState
from langfuse import get_client
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
langfuse = get_client()
class AgentDecision(BaseModel):
    thought: str  # brief reasoning (great for Langfuse!)
    next_action: Literal["generate_cypher", "write_final_answer"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_output_tokens=1024,
    google_api_key=settings.GEMINI_API_KEY,
)
AGENT_PROMPT = """You are the reasoning controller (the brain) of a Neo4j data agent.
Your ONLY job is to DECIDE the next step. You never write Cypher and you never
write the user's answer — other workers do that.

================================================
CURRENT SITUATION
================================================
User question : {user_query}
Rounds so far : {iteration_count} (maximum 4)
Generated query : {cypher_query}
Database error  : {error}
Database result : {node_results}

================================================
DECISION RULES (apply in order, stop at first match)
================================================
1. If there is no query and no error yet (nothing has been tried):
   → "generate_cypher"
2. If the database error starts with "UNANSWERABLE":
   → "write_final_answer"  (the data simply does not exist in this database)
3. If there is any other database error:
   → "generate_cypher"  (the query writer will see the error and fix the query)
4. If the result has rows:
   → "write_final_answer"
5. If the result is EMPTY and rounds < 3:
   → "generate_cypher"  (retry with a broader approach — the writer will be told to broaden)
6. If the result is EMPTY and rounds >= 3:
   → "write_final_answer"  (conclude honestly that no matching data exists)

Think briefly in "thought", then choose.
"""


def preview_results(rows, max_rows: int = 10) -> str:
    """The brain sees a TRUNCATED preview — never the full dump."""
    if rows is None:
        return "No results yet (the database has not been queried)."
    if not rows:
        return "EMPTY — the query ran successfully but returned zero rows."
    text = str(rows[:max_rows])
    if len(rows) > max_rows:
        text += f" ... ({len(rows) - max_rows} more rows)"
    return text[:2000]


async def agent_node(state: BrainAgentState) -> dict:
    prompt = AGENT_PROMPT.format(
        user_query=state["user_query"],
        iteration_count=state.get("iteration_count", 0),
        cypher_query=(state.get("cypher_query") or "— none yet —")[:300],
        error=state.get("error") or "None",
        node_results=preview_results(state.get("node_results")),
    )
    with langfuse.start_as_current_observation(
        as_type="generation", name="neo4j_main_agent"
    ) as gen:
        decision = await llm.with_structured_output(AgentDecision).ainvoke(prompt)

        gen.update(input=prompt, output=decision.model_dump_json())
    return {
        "next_action": decision.next_action,  # ← router reads this
        "iteration_count": state.get("iteration_count", 0)
        + 1,  # increment INSIDE the node
    }
