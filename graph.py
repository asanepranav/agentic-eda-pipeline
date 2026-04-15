"""
graph.py — Agentic EDA Pipeline using LangGraph + Groq

Architecture (StateGraph):
  START → data_profiler → stats_agent → viz_agent → insight_agent → report_agent → END

Each node reads from and writes to shared EDAState.
Follow-up chat uses the same state for context-aware Q&A.
"""

import io
import json
import warnings
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ── Shared State ──────────────────────────────────────────────────────────────
class EDAState(TypedDict):
    # Input
    df_json: str                              # DataFrame serialized as JSON
    filename: str                             # uploaded filename

    # Agent outputs
    profile: dict                             # shape, dtypes, nulls, duplicates
    stats: str                                # descriptive statistics summary
    viz_code: str                             # Python code to generate plots
    insights: str                             # LLM-generated insights
    report: str                               # final EDA report

    # Chat
    messages: Annotated[list, add_messages]   # conversation history
    chat_response: str                        # latest chat answer

    # Routing
    step: str                                 # current pipeline step


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm(api_key: str):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=api_key,
        max_tokens=2048,
    )


# ── Helper: deserialize DataFrame ────────────────────────────────────────────
def load_df(state: EDAState) -> pd.DataFrame:
    return pd.read_json(io.StringIO(state["df_json"]), orient="split")


# ── Node 1: Data Profiler ─────────────────────────────────────────────────────
def data_profiler_node(state: EDAState) -> EDAState:
    """
    Profiles the dataset — shape, dtypes, nulls, duplicates, cardinality.
    Pure pandas, no LLM needed.
    """
    df = load_df(state)

    null_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    cardinality = {col: int(df[col].nunique()) for col in df.columns}

    numeric_cols  = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    profile = {
        "shape":            {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns":          df.columns.tolist(),
        "dtypes":           {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_percentage":  null_pct,
        "duplicates":       int(df.duplicated().sum()),
        "cardinality":      cardinality,
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols":    datetime_cols,
        "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    return {**state, "profile": profile, "step": "profiled"}


# ── Node 2: Stats Agent ────────────────────────────────────────────────────────
def stats_agent_node(state: EDAState, api_key: str) -> EDAState:
    """
    Generates descriptive statistics and uses LLM to summarise key findings.
    """
    df      = load_df(state)
    profile = state["profile"]
    llm     = get_llm(api_key)

    # Compute stats
    numeric_stats = ""
    if profile["numeric_cols"]:
        desc = df[profile["numeric_cols"]].describe().round(3)
        numeric_stats = desc.to_string()

    cat_stats = ""
    if profile["categorical_cols"]:
        for col in profile["categorical_cols"][:5]:   # limit to 5
            top = df[col].value_counts().head(5)
            cat_stats += f"\n{col}:\n{top.to_string()}\n"

    # Correlation for numeric cols
    corr_str = ""
    if len(profile["numeric_cols"]) > 1:
        corr = df[profile["numeric_cols"]].corr().round(3)
        # Only strong correlations (>0.5 or <-0.5)
        strong = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong.append(f"{cols[i]} ↔ {cols[j]}: {val:.3f}")
        corr_str = "\n".join(strong) if strong else "No strong correlations found."

    prompt = f"""You are a data analyst. Summarise the key statistical findings from this dataset.

Dataset: {state['filename']}
Shape: {profile['shape']['rows']} rows × {profile['shape']['cols']} columns
Missing data: {json.dumps({k: v for k, v in profile['null_percentage'].items() if v > 0} or {'none': 0})}
Duplicates: {profile['duplicates']}

Numeric statistics:
{numeric_stats}

Top categorical values:
{cat_stats}

Strong correlations:
{corr_str}

Write a concise statistical summary (bullet points) covering:
- Data quality issues (nulls, duplicates)
- Key distributions and ranges
- Notable correlations
- Any potential outliers or anomalies
Keep it under 200 words."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {**state, "stats": response.content, "step": "stats_done"}


# ── Node 3: Visualization Agent ───────────────────────────────────────────────
def viz_agent_node(state: EDAState, api_key: str) -> EDAState:
    """
    Generates Python code for the most insightful visualizations.
    Code is executed in the Streamlit app using exec().
    """
    profile = state["profile"]
    llm     = get_llm(api_key)

    prompt = f"""You are a data visualization expert. Write Python code to create the most insightful EDA plots for this dataset.

Dataset info:
- Filename: {state['filename']}
- Numeric columns: {profile['numeric_cols']}
- Categorical columns: {profile['categorical_cols']}
- Shape: {profile['shape']['rows']} rows × {profile['shape']['cols']} columns

Write matplotlib/seaborn code. The DataFrame is available as variable `df`.
Create 4-6 plots in a grid. Use plt.tight_layout().

Rules:
- Import matplotlib.pyplot as plt and seaborn as sns at the top
- Use fig, axes = plt.subplots(...) for subplots
- Do NOT call plt.show() — the caller will handle that
- Handle cases where columns might not exist with try/except
- Use plt.style.use('seaborn-v0_8-whitegrid')
- Return ONLY the Python code, no explanation, no markdown backticks

Choose the most appropriate plots:
- Histograms or KDE for numeric distributions
- Bar charts for categorical counts
- Heatmap for correlations (if multiple numeric cols)
- Box plots for outlier detection
- Scatter plot for strongly correlated pairs"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Clean code — remove markdown if present
    code = response.content.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return {**state, "viz_code": code, "step": "viz_done"}


# ── Node 4: Insight Agent ─────────────────────────────────────────────────────
def insight_agent_node(state: EDAState, api_key: str) -> EDAState:
    """
    Synthesises deep business/analytical insights from profile + stats.
    """
    llm = get_llm(api_key)

    prompt = f"""You are a senior data scientist. Based on the EDA findings below, generate actionable insights.

Dataset: {state['filename']}
Profile: {json.dumps(state['profile'], indent=2)}

Statistical summary:
{state['stats']}

Generate 5-7 specific, actionable insights covering:
1. Data quality recommendations (what to clean/fix before modeling)
2. Feature engineering opportunities
3. Potential target variables if this looks like a modeling problem
4. Business observations from distributions and correlations
5. Columns to drop or transform
6. Suggested next steps for analysis

Be specific — reference actual column names and values. Format as numbered list."""

    response = llm.invoke([
        SystemMessage(content="You are an expert data scientist who gives precise, actionable EDA insights."),
        HumanMessage(content=prompt)
    ])

    return {**state, "insights": response.content, "step": "insights_done"}


# ── Node 5: Report Agent ──────────────────────────────────────────────────────
def report_agent_node(state: EDAState, api_key: str) -> EDAState:
    """
    Compiles everything into a final structured EDA report.
    """
    llm = get_llm(api_key)

    prompt = f"""You are a technical writer. Compile a complete EDA report from the findings below.

Dataset: {state['filename']}
Shape: {state['profile']['shape']['rows']} rows × {state['profile']['shape']['cols']} columns

Statistical Summary:
{state['stats']}

Key Insights:
{state['insights']}

Write a professional EDA report with these sections:
1. Executive Summary (3-4 sentences)
2. Dataset Overview (shape, columns, types)
3. Data Quality Assessment (nulls, duplicates, issues)
4. Statistical Highlights (key distributions, correlations)
5. Key Findings & Insights (top 5 findings)
6. Recommendations (what to do next)

Use clear headings. Keep it professional and concise. Around 400-500 words."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {**state, "report": response.content, "step": "report_done"}


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_eda_graph(api_key: str):
    """Build and compile the LangGraph EDA pipeline."""

    graph = StateGraph(EDAState)

    # Add nodes — wrap stateful nodes with api_key via lambda
    graph.add_node("data_profiler",  data_profiler_node)
    graph.add_node("stats_agent",    lambda s: stats_agent_node(s, api_key))
    graph.add_node("viz_agent",      lambda s: viz_agent_node(s, api_key))
    graph.add_node("insight_agent",  lambda s: insight_agent_node(s, api_key))
    graph.add_node("report_agent",   lambda s: report_agent_node(s, api_key))

    # Linear pipeline
    graph.set_entry_point("data_profiler")
    graph.add_edge("data_profiler", "stats_agent")
    graph.add_edge("stats_agent",   "viz_agent")
    graph.add_edge("viz_agent",     "insight_agent")
    graph.add_edge("insight_agent", "report_agent")
    graph.add_edge("report_agent",  END)

    return graph.compile()


# ── Run Pipeline ──────────────────────────────────────────────────────────────
def run_eda_pipeline(df: pd.DataFrame, filename: str, api_key: str) -> dict:
    """
    Run the full EDA pipeline on a DataFrame.
    Returns final state with profile, stats, viz_code, insights, report.
    """
    app = build_eda_graph(api_key)

    initial_state = EDAState(
        df_json=df.to_json(orient="split"),
        filename=filename,
        profile={},
        stats="",
        viz_code="",
        insights="",
        report="",
        messages=[],
        chat_response="",
        step="start",
    )

    steps = []
    final_state = None

    for event in app.stream(initial_state, {"recursion_limit": 20}):
        for node_name, node_state in event.items():
            steps.append(node_name)
            final_state = node_state

    final_state["_steps"] = steps
    return final_state


# ── Chat with Data ────────────────────────────────────────────────────────────
def chat_with_data(question: str, eda_state: dict, api_key: str) -> str:
    """
    Answer follow-up questions using EDA context + pandas query execution.
    """
    llm = get_llm(api_key)
    df  = pd.read_json(io.StringIO(eda_state["df_json"]), orient="split")

    # Try to answer with pandas first, fall back to LLM
    system = f"""You are a data analyst assistant. You have full knowledge of this dataset.

Dataset: {eda_state['filename']}
Shape: {eda_state['profile']['shape']['rows']} rows × {eda_state['profile']['shape']['cols']} columns
Columns: {eda_state['profile']['columns']}
Numeric columns: {eda_state['profile']['numeric_cols']}
Categorical columns: {eda_state['profile']['categorical_cols']}

EDA Summary:
{eda_state['stats']}

Key Insights:
{eda_state['insights']}

Answer the user's question concisely and precisely.
If the question requires a calculation, provide the exact number.
If it requires a comparison, compare specifically.
Always reference actual column names."""

    # Build message history
    history = eda_state.get("messages", [])[-6:]  # last 3 turns
    messages = [SystemMessage(content=system)] + history + [HumanMessage(content=question)]

    response = llm.invoke(messages)
    return response.content
