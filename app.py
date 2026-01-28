"""Streamlit Chat with CrewAI Agents + Exa MCP + DeepEval Evaluation."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

st.set_page_config(page_title="Cloud Lease Negotiator", page_icon="☁️", layout="wide")

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_ground_truth():
    vm_path = DATA_DIR / "cloud_cluster_dataset.csv"
    if not vm_path.exists():
        return None
    try:
        df = pd.read_csv(vm_path)
        zombie_mask = (df['avg_cpu_usage_percent'] < 30) & (df['avg_ram_usage_percent'] < 30)
        zombies = df[zombie_mask]
        return {
            "zombie_count": len(zombies),
            "total_vms": len(df),
            "wasted_cost": zombies['monthly_cost_usd'].sum()
        }
    except Exception:
        return None


def clean_agent_response(response: str) -> str:
    """Remove LLM thinking/action patterns from agent response."""
    import re
    
    # Remove "Action: tool_name" lines
    response = re.sub(r'Action:\s*\w+\s*\n?', '', response)
    
    # Remove "Action Input: {...}" blocks
    response = re.sub(r'Action Input:\s*\{[^}]*\}\s*\n?', '', response)
    
    # Remove "Thought:" lines
    response = re.sub(r'Thought:\s*[^\n]*\n?', '', response)
    
    # Remove "Observation:" lines
    response = re.sub(r'Observation:\s*[^\n]*\n?', '', response)
    
    # Remove "Final Answer:" prefix if present
    response = re.sub(r'^Final Answer:\s*', '', response, flags=re.MULTILINE)
    
    # Clean up multiple newlines
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    return response.strip()


def run_crew_analysis(user_query: str) -> tuple[str, bool, list]:
    tools_used = []
    try:
        from src.crew import create_chat_crew
        
        needs_search = any(kw in user_query.lower() for kw in [
            'pricing', 'price', 'search', 'web', 'current', 'market'
        ]) and os.getenv("EXA_API_KEY")
        
        crew = create_chat_crew(user_query, use_web_search=needs_search)
        result = crew.kickoff()
        
        tools_used = ["filter_underutilized_vms"]
        if any(kw in user_query.lower() for kw in ['savings', 'calculate', 'total', 'roi']):
            tools_used.extend(["batch_analyze_and_log", "calculate_total_savings"])
        if needs_search:
            tools_used.append("exa_web_search")
        
        if hasattr(result, 'raw'):
            raw_response = result.raw
        elif hasattr(result, 'output'):
            raw_response = result.output
        else:
            raw_response = str(result)
        
        # Clean the response to remove thinking/action patterns
        clean_response = clean_agent_response(raw_response)
        return clean_response, needs_search, tools_used
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```", False, tools_used


def run_deepeval(query: str, response: str, tools_used: list) -> dict:
    try:
        from src.evaluation.deepeval_integration import evaluate_with_deepeval, DEEPEVAL_AVAILABLE
        if not DEEPEVAL_AVAILABLE:
            return {"error": "DeepEval not installed"}
        return evaluate_with_deepeval(query, response, tools_used)
    except Exception as e:
        return {"error": str(e)}


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "evals" not in st.session_state:
    st.session_state.evals = []
if "show_eval" not in st.session_state:
    st.session_state.show_eval = False

# Sidebar
with st.sidebar:
    # Status Section
    st.markdown("### 📡 Status")
    
    if os.getenv("GROQ_API_KEY"):
        st.success("✓ Groq API")
    else:
        st.error("✗ GROQ_API_KEY missing")
    
    if os.getenv("EXA_API_KEY"):
        st.success("✓ Exa API (MCP)")
    else:
        st.warning("○ EXA_API_KEY missing")
    
    # Check CrewAI agents
    try:
        from src.crew import create_chat_crew
        st.success("✓ CrewAI Agents")
        st.caption("Auditor → Architect → CFO")
    except:
        st.error("✗ CrewAI error")
    
    st.divider()
    
    # Evaluation Section
    st.markdown("### 📊 Evaluation")
    st.session_state.show_eval = st.toggle("Enable DeepEval", value=st.session_state.show_eval, help="Enable LLM-as-judge evaluation metrics")
    
    if st.session_state.show_eval:
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            st.success("✓ Gemini API")
        else:
            st.warning("○ GOOGLE_API_KEY needed")
        
        try:
            from src.evaluation.deepeval_integration import DEEPEVAL_AVAILABLE
            if DEEPEVAL_AVAILABLE:
                st.success("✓ DeepEval")
            else:
                st.warning("○ DeepEval not installed")
        except:
            st.warning("○ DeepEval error")
    
    st.divider()
    
    # Data Section
    st.markdown("### 📁 Data")
    st.markdown("**Upload VM CSV**")
    vm_file = st.file_uploader("Drag and drop file here", type=["csv"], key="vm_upload", label_visibility="collapsed")
    if vm_file:
        with open(DATA_DIR / "cloud_cluster_dataset.csv", "wb") as f:
            f.write(vm_file.getbuffer())
        st.success("Uploaded!")
    
    stats = get_ground_truth()
    if stats:
        st.divider()
        st.metric("Total VMs", f"{stats['total_vms']:,}")
        st.metric("Zombie Instances", f"{stats['zombie_count']:,}")
        st.metric("Wasted/Month", f"${stats['wasted_cost']:,.0f}")
    
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.evals = []
        st.rerun()

# Main
st.title("☁️ Cloud Lease Negotiator")

if not st.session_state.messages:
    st.markdown("### Welcome! 👋 ☁️")
    st.markdown("I analyze your cloud infrastructure with **CrewAI Agents**:")
    st.markdown("""
- 🔍 **Auditor** - Finds zombie VMs
- 🏗️ **Architect** - Recommends rightsizing
- 💰 **CFO** - Calculates savings
""")
    st.markdown('**Try:** "How many zombie instances?" or "Calculate total savings"')

# Messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", "\\$"))
        
        if msg["role"] == "assistant" and st.session_state.show_eval:
            eval_idx = i // 2
            if eval_idx < len(st.session_state.evals):
                ev = st.session_state.evals[eval_idx]
                if ev and "error" not in ev:
                    with st.expander("📊 DeepEval Metrics"):
                        res = ev.get('results', {})
                        c1, c2 = st.columns(2)
                        
                        task = res.get('task_completion', {})
                        c1.metric("Task Completion", f"{task.get('score', 0):.0%}", 
                                  delta="Pass" if task.get('passed') else "Fail")
                        
                        tool = res.get('tool_correctness', {})
                        c2.metric("Tool Correctness", f"{tool.get('score', 0):.0%}", 
                                  delta="Pass" if tool.get('passed') else "Fail")
                        
                        st.progress(ev.get('overall_score', 0), text=f"Overall: {ev.get('overall_score', 0):.0%}")
                elif ev and "error" in ev:
                    with st.expander("📊 DeepEval Metrics"):
                        st.error(ev['error'])

# Input
if prompt := st.chat_input("Ask about your cloud infrastructure..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        vm_path = DATA_DIR / "cloud_cluster_dataset.csv"
        
        if not vm_path.exists():
            response = "Please upload VM data first."
            tools_used = []
        elif not os.getenv("GROQ_API_KEY"):
            response = "GROQ_API_KEY not set"
            tools_used = []
        else:
            with st.spinner("🔄 Running CrewAI Agents..."):
                response, used_mcp, tools_used = run_crew_analysis(prompt)
            if used_mcp:
                st.info("🌐 Used Exa MCP for web search")
        
        st.markdown(response.replace("$", "\\$"))
        
        # DeepEval
        if tools_used and st.session_state.show_eval:
            with st.spinner("📊 Running DeepEval..."):
                ev = run_deepeval(prompt, response, tools_used)
            st.session_state.evals.append(ev)
            
            if "error" not in ev:
                with st.expander("📊 DeepEval Metrics", expanded=True):
                    res = ev.get('results', {})
                    c1, c2 = st.columns(2)
                    
                    task = res.get('task_completion', {})
                    c1.metric("Task Completion", f"{task.get('score', 0):.0%}",
                              delta="Pass" if task.get('passed') else "Fail")
                    if task.get('reason'):
                        c1.caption(task['reason'][:150] + "..." if len(task.get('reason', '')) > 150 else task.get('reason', ''))
                    
                    tool = res.get('tool_correctness', {})
                    c2.metric("Tool Correctness", f"{tool.get('score', 0):.0%}",
                              delta="Pass" if tool.get('passed') else "Fail")
                    if tool.get('reason'):
                        c2.caption(tool['reason'][:150] + "..." if len(tool.get('reason', '')) > 150 else tool.get('reason', ''))
                    
                    st.progress(ev.get('overall_score', 0), text=f"Overall: {ev.get('overall_score', 0):.0%}")
            else:
                with st.expander("📊 DeepEval Metrics", expanded=True):
                    st.error(ev['error'])
        else:
            st.session_state.evals.append({})
        
        st.session_state.messages.append({"role": "assistant", "content": response})
