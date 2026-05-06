from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.agent.state import AgentState
from src.llm.factory import get_llm
from src.prompts.few_shot import FEW_SHOT_EXAMPLES
from src.prompts.system_prompts import GENERATE_SYSTEM_PROMPT
from src.schema.loader import get_schema_for_query


def generate_sql(state: AgentState, config: RunnableConfig) -> dict:
    print("--- NODE: GENERATING SQL ---")
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "gpt-4o-mini")
    provider = configurable.get("provider", "openai")
    use_few_shot = configurable.get("use_few_shot", True)
    use_rag = configurable.get("use_rag", True)

    question = state.get("standalone_query")
    error = state.get("error", "")
    schema = get_schema_for_query(question, use_rag=use_rag)

    llm = get_llm(provider=provider, model_name=model_name, temperature=0)
    system_prompt = GENERATE_SYSTEM_PROMPT.format(schema=schema)
    if use_few_shot:
        system_prompt += f"\n{FEW_SHOT_EXAMPLES}\n"
    if error:
        system_prompt += f"\n\nWARNING: Your previous query failed with this error: {error}. Please correct your SQL."
    plan = state.get("sql_plan", "")
    if plan:
        system_prompt += f"\n\nCRITICAL: Follow this architectural plan to write your SQL:\n{plan}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    response = llm.invoke(messages)
    generated_sql = response.content.strip()
    print(f"Generated SQL:\n{generated_sql}\n")
    return {"sql_query": generated_sql}
