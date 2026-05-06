from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.agent.state import AgentState
from src.llm.factory import get_llm
from src.prompts.system_prompts import PLAN_SYSTEM_PROMPT
from src.schema.loader import get_schema_for_query


def plan_sql_query(state: AgentState, config: RunnableConfig) -> dict:
    print("--- NODE: PLANNING COMPLEX SQL ---")
    query = state.get("standalone_query")
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "gpt-4o-mini")
    provider = configurable.get("provider", "openai")
    use_rag = configurable.get("use_rag", True)
    schema = get_schema_for_query(query, use_rag=use_rag)
    llm = get_llm(provider=provider, model_name=model_name, temperature=0)
    system_prompt = PLAN_SYSTEM_PROMPT.format(schema=schema)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Plan the SQL execution for this question: {query}")
    ])
    plan = response.content.strip()
    print("Plan generated.\n")
    return {"sql_plan": plan}
