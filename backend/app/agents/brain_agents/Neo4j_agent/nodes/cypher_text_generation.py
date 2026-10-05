import os
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from app.core.config import settings
from langfuse import get_client, observe
from langchain.tools import tool
from neo4j import AsyncGraphDatabase
from app.agents.brain_agents.Neo4j_agent.state import BrainAgentState
from app.agents.brain_agents.Neo4j_agent.neo4j_main_node import (
    preview_results,
)

# Langfuse observability is intentionally not checked during import. A remote
# auth request here made FastAPI startup fail when Langfuse was not configured,
# even though the Neo4j endpoint was not being used.
langfuse = get_client()
if not all((settings.NEO4J_URI, settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)):
    raise ValueError(
        "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in .env"
    )

URI = settings.NEO4J_URI
AUTH = (settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
driver = AsyncGraphDatabase.driver(URI, auth=AUTH)
# model initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_output_tokens=8192,
    thinking_budget=0,
    google_api_key=settings.GEMINI_API_KEY,
)
PROMPT = """You are a Neo4j Cypher Query Generator. Your ONLY job is to convert a user's 
natural language question into a single, valid, executable, read-only Cypher 
query for the graph database described below.

========================================
DATABASE SCHEMA
========================================

NODE LABELS & PROPERTIES:

- Department       { deptId, name }
- Employee         { empId, employee_id, name, education, experience, location }
- Job              { jobId, title, jd_text, jobLevel, department }
- KRA              { kraId, name }
- KPI              { kpiId, name, metric, frequency, target, 
                     thresholdBelow, thresholdMeets, thresholdExcellent }
- Responsibility   { respId, text }
- Skill            { skillId, name }
- Tool             { toolId, name }
- jd_sessions      { id, title, purpose, style, jd_text, data, nodes, 
                     relationships, visualisation, sent_to_hr_at }

RELATIONSHIPS:
- (Department)-[:EMPLOYS]->(Employee)
- (Employee)-[:HAS_ROLE]->(Job)
- (Employee)-[:REPORTS_TO]->(Employee)
- (Job)-[:REQUIRES_SKILL]->(Skill)
- (Job)-[:REQUIRES_TOOL]->(Tool)
- (Job)-[:HAS_KRA]->(KRA)
- (Job)-[:HAS_RESPONSIBILITY]->(Responsibility)
- (KRA)-[:MEASURED_BY]->(KPI)

IMPORTANT PROPERTY NOTES:
- Skill, Tool, KRA, Responsibility are identified by BOTH an id 
  (skillId / toolId / kraId / respId) and a human-readable name or text. 
  For user-facing questions, always match or return the human-readable 
  field (name / text), NOT the id.
- Employee has two id fields: empId and employee_id. Prefer matching 
  employees by name unless the user gives an id.
- KPI performance levels: thresholdBelow, thresholdMeets, thresholdExcellent 
  describe rating bands for the KPI's metric.
- jd_sessions stores generated JD session outputs (data, nodes, 
  relationships, visualisation are large serialized fields). NEVER return 
  these fields unless the user explicitly asks for session data. If needed, 
  return only id, title, purpose, style, sent_to_hr_at.
- Job is best matched by its `title` property; use jd_text only when the 
  user asks about job description content.

========================================
RULES
========================================
1. Output ONLY the Cypher query. No explanations, no markdown, no code fences.
2. Use ONLY the node labels, relationship types, and property names listed 
   above. Never invent or assume others.
3. Text matching must be case-insensitive:
   WHERE toLower(n.name) CONTAINS toLower('<value>')
4. WHERE PLACEMENT IS CRITICAL:
- A WHERE clause always filters the MATCH or OPTIONAL MATCH directly above it.
- To filter the primary entity (Employee, Department, Job...), place WHERE 
    IMMEDIATELY AFTER the first MATCH — NEVER after an OPTIONAL MATCH.
- WRONG:  MATCH (a)-[:R]->(b) OPTIONAL MATCH (b)-[:S]->(c) WHERE a.name CONTAINS 'x'
- RIGHT:  MATCH (a)-[:R]->(b) WHERE a.name CONTAINS 'x' OPTIONAL MATCH (b)-[:S]->(c)
5. Use RETURN DISTINCT when duplicates are possible.
6. Use clear aliases in RETURN (e.g., AS Skill, AS Employee, AS KPI).
7. Add LIMIT 25 for broad/exploratory questions ("list", "show me", "what are").
8. Use count() or collect() for "how many" / "which all" questions.
9. Respect relationship directions exactly as defined in the schema.
10. Read-only queries only: MATCH / OPTIONAL MATCH / WHERE / RETURN / ORDER BY.
   Never use CREATE, MERGE, SET, DELETE, DETACH DELETE, REMOVE, or CALL that 
   writes.
11. If the question cannot be answered with this schema, return exactly:
    // ERROR: Cannot generate query for this question with current schema
12. Single-line or cleanly formatted multi-line Cypher only. No comments 
    inside the query.

========================================
EXAMPLES
========================================

Q: What skills are required for jobs in the accounts department?
MATCH (d:Department)-[:EMPLOYS]->(e:Employee)-[:HAS_ROLE]->(j:Job)-[:REQUIRES_SKILL]->(s:Skill)
WHERE toLower(d.name) CONTAINS 'accounts'
RETURN DISTINCT s.name AS Skill

Q: Which jobs require the ERP / Corporate Management Systems tool?
MATCH (j:Job)-[:REQUIRES_TOOL]->(t:Tool)
WHERE toLower(t.name) CONTAINS 'erp'
RETURN DISTINCT j.title AS Job

Q: What are the KRAs, their responsibilities, and KPIs for a job?
MATCH (j:Job)-[:HAS_KRA]->(kra:KRA)
WHERE toLower(j.title) CONTAINS toLower('accounts executive')
OPTIONAL MATCH (kra)-[:HAS_RESPONSIBILITY]->(r:Responsibility)
OPTIONAL MATCH (kra)-[:MEASURED_BY]->(kpi:KPI)
RETURN kra.name AS KRA, collect(DISTINCT r.text) AS Responsibilities,
       collect(DISTINCT kpi.name + ' (' + kpi.metric + ')') AS KPIs

Q: What is the target and threshold for a specific KPI?
MATCH (kra:KRA)-[:MEASURED_BY]->(kpi:KPI)
WHERE toLower(kpi.name) CONTAINS toLower('<kpi name>')
RETURN kpi.name AS KPI, kpi.metric AS Metric, kpi.target AS Target,
       kpi.thresholdBelow AS Below, kpi.thresholdMeets AS Meets,
       kpi.thresholdExcellent AS Excellent

Q: How many employees are in each department?
MATCH (d:Department)-[:EMPLOYS]->(e:Employee)
RETURN d.name AS Department, count(e) AS EmployeeCount
ORDER BY EmployeeCount DESC

Q: Who reports to a given manager?
MATCH (report:Employee)-[:REPORTS_TO]->(mgr:Employee)
WHERE toLower(mgr.name) CONTAINS toLower('<name>')
RETURN report.name AS DirectReport

Q: List recent JD sessions sent to HR.
MATCH (s:jd_sessions)
WHERE s.sent_to_hr_at IS NOT NULL
RETURN s.id, s.title, s.sent_to_hr_at
ORDER BY s.sent_to_hr_at DESC LIMIT 25

Now generate the Cypher query for the user's question below.

User Question: {user_query}"""


async def get_cypher_query(state: BrainAgentState) -> dict:
    prompt = PROMPT.replace("{user_query}", state["user_query"])

    # ---- CORRECTION BLOCK 1: the query blew up ----
    if state.get("error") and not state["error"].startswith("UNANSWERABLE"):
        prompt += f"""

=== PREVIOUS ATTEMPT FAILED ===
Your previous query:
{state.get('cypher_query') or '(none)'}

Database error:
{state['error']}

Fix the problem and return ONE corrected read-only Cypher query only.
If the error says a label or property does not exist, re-read the schema
above and use only valid names.
"""

    # ---- CORRECTION BLOCK 2: ran fine, found nothing ----
    elif state.get("node_results") is not None and len(state["node_results"]) == 0:
        prompt += """

=== PREVIOUS ATTEMPT RETURNED ZERO ROWS ===
The query executed successfully but matched nothing.
Broaden the approach: verify label and property spellings against the schema,
prefer CONTAINS over exact equality, drop overly restrictive WHERE filters.
Return ONE corrected read-only Cypher query only.Improve the query quality check in depth into the database not overall view search for roles, skills, tools, KRAs, KPIs, and reporting lines. Ensure the query is optimized for performance and accuracy.

"""

    try:
        with langfuse.start_as_current_observation(
            as_type="generation", name="cypher_generator"
        ) as gen:
            response = await llm.ainvoke(prompt)
            gen.update(input=prompt, output=response.content)  
        content = (response.content or "").strip()

        # ---- your prompt's rule 11 → convert to an UNANSWERABLE signal ----
        if content.startswith("// ERROR:"):
            return {
                "error": "UNANSWERABLE: " + content.removeprefix("// ERROR:").strip(),
                "cypher_query": "",
            }
        if not content:
            return {"cypher_query": "", "error": "LLM returned an empty query."}
        print(f"Generated Cypher query:\n{content}")
        return {"cypher_query": content, "error": None}
    except Exception as e:
        return {"cypher_query": "", "error": f"Query generation failed: {e}"}


async def execute_cypher_node(state: BrainAgentState) -> dict:
    try:
        async with driver.session(default_access_mode="READ") as session:
            result = await session.run(state["cypher_query"])
            rows = await result.data()
            print(f"Query results:\n{rows}")
        return {"node_results": rows, "error": None}
    except Exception as e:
        print(f"Cypher execution failed: {e}")
        return {"node_results": [], "error": str(e)}


FINAL_ANSWER_PROMPT = """You are the answer writer of a data agent.

User question : {user_query}
Situation     : {situation}
Data          : {node_results}

RULES
- Use ONLY the data above. Never invent names, numbers, or facts.
- If the situation says no data was found, say that clearly and briefly —
  never fill the gap with plausible-sounding guesses.
- If the situation says UNANSWERABLE, politely explain you can't answer that
  from the available data (departments, employees, jobs, skills, tools,
  KRAs, KPIs, reporting lines).
- Format for a human: short sentences; bullet list or small table for
  multiple items.
- Don't mention Cypher, queries, or "the database" unless the user asked.
"""


async def final_answer(state: BrainAgentState) -> dict:
    err = state.get("error") or ""
    rows = state.get("node_results")
    if err.startswith("UNANSWERABLE"):
        situation = (
            "UNANSWERABLE — this question cannot be answered from the available data."
        )
    elif err:
        situation = "FAILED — the data could not be retrieved after several attempts. Be honest about it."
    elif not rows:
        situation = "NO DATA — the queries ran but found nothing matching."
    else:
        situation = "VERIFIED — the data below came directly from the database."
    prompt = FINAL_ANSWER_PROMPT.format(
        user_query=state["user_query"],
        situation=situation,
        node_results=preview_results(rows, max_rows=30),
    )
    with langfuse.start_as_current_observation(
        as_type="generation", name="final_answer"
    ) as gen:
        resp = await llm.ainvoke(prompt)
        print(f"Final answer:\n{resp.content}")
        gen.update(input=prompt, output=resp.content)
    return {"final_result": resp.content}
