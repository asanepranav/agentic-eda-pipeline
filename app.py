"""
app.py — Agentic EDA Pipeline
Upload any CSV → agents auto-analyse → get report + chat with your data
"""

import io
import os
import traceback
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from graph import run_eda_pipeline, chat_with_data
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Agentic EDA", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #07090f; color: #e8e6df; }

    .agent-step {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 14px; border-radius: 6px;
        background: #111827; margin: 4px 0;
        font-size: 0.82rem; font-family: 'Space Mono', monospace;
    }
    .step-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
    .dot-profiler { background:#06b6d4; }
    .dot-stats    { background:#8b5cf6; }
    .dot-viz      { background:#f59e0b; }
    .dot-insight  { background:#10b981; }
    .dot-report   { background:#f43f5e; }

    .report-box {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 12px; padding: 24px;
        line-height: 1.9; color: #d1d5db; white-space: pre-wrap;
        font-size: 0.9rem;
    }
    .insight-box {
        background: #0d1117; border-left: 3px solid #10b981;
        border-radius: 0 8px 8px 0; padding: 16px 20px;
        color: #d1d5db; white-space: pre-wrap; font-size: 0.88rem;
    }
    .chat-user {
        background: #1e1e28; border-left: 3px solid #8b5cf6;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin: 6px 0; font-size: 0.88rem;
    }
    .chat-bot {
        background: #111827; border-left: 3px solid #06b6d4;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin: 6px 0; font-size: 0.88rem; color: #c8d8ea;
    }
    .stButton > button {
        background: #8b5cf6 !important; color: #fff !important;
        font-weight: 700 !important; border: none !important;
        border-radius: 8px !important; width: 100% !important;
        padding: 0.65rem !important;
        font-family: 'Space Mono', monospace !important;
    }
    div[data-testid="stSidebar"] { background: #050709; border-right: 1px solid #1f2937; }
    .stTextInput > div > div > input, .stTextArea textarea {
        background: #111827 !important; color: #e8e6df !important;
        border: 1px solid #374151 !important; border-radius: 8px !important;
    }
    .metric-card {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 8px; padding: 14px 16px; text-align: center;
    }
    .metric-num { font-family:'Space Mono',monospace; font-size:1.5rem; color:#8b5cf6; font-weight:700; }
    .metric-lbl { font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.06em; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("eda_state", None), ("chat_history", []), ("df", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#8b5cf6;margin-bottom:0.4rem">Agentic EDA</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.75rem;color:#6b7280;margin-bottom:1.5rem">LangGraph · Groq · Auto-analysis</div>', unsafe_allow_html=True)

    api_key = st.text_input("Groq API Key", type="password",
                             value=os.environ.get("GROQ_API_KEY", ""),
                             placeholder="gsk_...")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✓ {uploaded.name} loaded")
            st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.session_state.df is not None and api_key:
        if st.button("🚀 Run EDA Pipeline"):
            with st.spinner("Agents running... ~1-2 minutes"):
                try:
                    result = run_eda_pipeline(
                        st.session_state.df,
                        uploaded.name if uploaded else "dataset.csv",
                        api_key
                    )
                    st.session_state.eda_state   = result
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error: {str(e)}")
    elif st.session_state.df is not None and not api_key:
        st.warning("Enter Groq API key above.")

    if st.session_state.eda_state:
        st.divider()
        if st.button("🗑 Reset"):
            st.session_state.eda_state   = None
            st.session_state.chat_history = []
            st.session_state.df          = None
            st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem;color:#374151;font-family:'Space Mono',monospace;line-height:2.2">
    pipeline nodes<br>
    <span style="color:#06b6d4">■</span> data profiler<br>
    <span style="color:#8b5cf6">■</span> stats agent<br>
    <span style="color:#f59e0b">■</span> viz agent<br>
    <span style="color:#10b981">■</span> insight agent<br>
    <span style="color:#f43f5e">■</span> report agent
    </div>
    """, unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.7rem;font-weight:700;color:#8b5cf6">🔍 Agentic EDA Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#6b7280;margin-bottom:1.5rem;font-size:0.85rem">Upload a CSV → LangGraph agents auto-profile, analyse, visualise, and report</div>', unsafe_allow_html=True)

if not st.session_state.eda_state:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:50vh;gap:12px;text-align:center">
        <div style="font-size:2.5rem">🔍</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#8b5cf6">
            Upload a CSV to begin
        </div>
        <div style="color:#4b5563;font-size:0.85rem;max-width:420px;line-height:1.8">
            5 specialised agents will automatically profile your data,<br>
            compute statistics, generate visualisations, extract insights,<br>
            and compile a full EDA report. Then chat with your data.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    state = st.session_state.eda_state
    profile = state.get("profile", {})

    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    shape = profile.get("shape", {})
    nulls = sum(1 for v in profile.get("null_percentage", {}).values() if v > 0)
    m1.markdown(f'<div class="metric-card"><div class="metric-num">{shape.get("rows",0):,}</div><div class="metric-lbl">Rows</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-num">{shape.get("cols",0)}</div><div class="metric-lbl">Columns</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-num">{len(profile.get("numeric_cols",[]))}</div><div class="metric-lbl">Numeric</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-num">{nulls}</div><div class="metric-lbl">Cols with nulls</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="metric-card"><div class="metric-num">{profile.get("duplicates",0)}</div><div class="metric-lbl">Duplicates</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Agent execution trace ──────────────────────────────────────────────────
    steps_done = state.get("_steps", [])
    step_map = {
        "data_profiler": ("Data profiler", "dot-profiler", "Profiled dataset structure"),
        "stats_agent":   ("Stats agent",   "dot-stats",    "Computed descriptive statistics"),
        "viz_agent":     ("Viz agent",     "dot-viz",      "Generated visualization code"),
        "insight_agent": ("Insight agent", "dot-insight",  "Extracted analytical insights"),
        "report_agent":  ("Report agent",  "dot-report",   "Compiled final report"),
    }
    with st.expander("Agent execution trace", expanded=False):
        for step in steps_done:
            if step in step_map:
                label, dot_cls, desc = step_map[step]
                st.markdown(f'<div class="agent-step"><div class="step-dot {dot_cls}"></div><strong>{label}</strong> — {desc}</div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualisations", "📋 Report", "💡 Insights", "🔬 Data Profile", "💬 Chat"])

    with tab1:
        st.markdown("### Auto-generated visualisations")
        viz_code = state.get("viz_code", "")
        if viz_code:
            try:
                df = pd.read_json(io.StringIO(state["df_json"]), orient="split")
                exec_globals = {"df": df, "plt": plt, "pd": pd}
                import seaborn as sns
                exec_globals["sns"] = sns
                import numpy as np
                exec_globals["np"] = np
                exec(viz_code, exec_globals)
                st.pyplot(plt.gcf())
                plt.close("all")
            except Exception as e:
                st.warning(f"Visualisation error: {e}")
                st.code(viz_code, language="python")
        else:
            st.info("No visualisation code generated.")

        with st.expander("View generated plot code"):
            st.code(viz_code, language="python")

    with tab2:
        st.markdown("### EDA Report")
        st.markdown(f'<div class="report-box">{state.get("report","")}</div>', unsafe_allow_html=True)
        st.download_button(
            "📥 Download report",
            data=state.get("report", ""),
            file_name=f"eda_report_{state.get('filename','data').replace('.csv','')}.txt",
            mime="text/plain"
        )

    with tab3:
        st.markdown("### Agent insights")
        st.markdown(f'<div class="insight-box">{state.get("insights","")}</div>', unsafe_allow_html=True)
        st.markdown("### Statistical summary")
        st.markdown(f'<div class="insight-box">{state.get("stats","")}</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown("### Dataset profile")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column types**")
            dtypes_df = pd.DataFrame({
                "Column": list(profile.get("dtypes", {}).keys()),
                "Type":   list(profile.get("dtypes", {}).values()),
                "Null %": [profile.get("null_percentage", {}).get(c, 0) for c in profile.get("dtypes", {}).keys()],
                "Unique": [profile.get("cardinality", {}).get(c, 0) for c in profile.get("dtypes", {}).keys()],
            })
            st.dataframe(dtypes_df, use_container_width=True)
        with col2:
            st.markdown("**Dataset preview**")
            try:
                df_preview = pd.read_json(io.StringIO(state["df_json"]), orient="split")
                st.dataframe(df_preview.head(10), use_container_width=True)
            except Exception:
                pass

    with tab5:
        st.markdown("### Chat with your data")
        st.markdown('<div style="font-size:0.8rem;color:#6b7280;margin-bottom:1rem">Ask anything about your dataset — the agent has full context from the EDA.</div>', unsafe_allow_html=True)

        # Quick questions
        quick_qs = [
            "What are the top 3 most important features?",
            "Which columns have the most missing data?",
            "What correlations should I investigate further?",
            "What should I do before training an ML model?",
        ]
        qcols = st.columns(2)
        selected_q = None
        for i, q in enumerate(quick_qs):
            if qcols[i % 2].button(q, key=f"qq_{i}"):
                selected_q = q

        # Chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        question = st.text_input("Ask a question", value=selected_q or "", placeholder="e.g. What is the average value of column X?", key="chat_input")
        if st.button("Ask →", key="ask_btn") and question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                try:
                    # Add message history to state for context
                    state_with_history = {
                        **state,
                        "messages": [
                            HumanMessage(content=m["content"]) if m["role"] == "user"
                            else AIMessage(content=m["content"])
                            for m in st.session_state.chat_history[:-1]
                        ]
                    }
                    answer = chat_with_data(question, state_with_history, api_key)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.rerun()
