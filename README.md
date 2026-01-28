# Cloud Lease Negotiator

CrewAI multi-agent system for VM right-sizing with **Exa MCP** for live pricing search and **DeepEval** for agent evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Chat UI                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   3-Agent Pipeline                          │
│  ┌──────────┐    ┌───────────┐    ┌─────────┐              │
│  │ Auditor  │───▶│ Architect │───▶│   CFO   │              │
│  │(Filter)  │    │(LLM Sense)│    │(Savings)│              │
│  └──────────┘    └─────┬─────┘    └─────────┘              │
│                        │                                    │
│                        ▼                                    │
│              ┌─────────────────┐                           │
│              │    Exa MCP      │                           │
│              │ (Live Pricing)  │                           │
│              └─────────────────┘                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  DeepEval Metrics                           │
│  ┌────────────────┐  ┌──────────────────┐                  │
│  │Task Completion │  │ Tool Correctness │                  │
│  └────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## DeepEval Integration (Gemini-powered)

Evaluates agent performance using Gemini as LLM judge with three metrics:

### TaskCompletionMetric
Evaluates how effectively the agent accomplishes the user's task by analyzing the trace and comparing outcome against the inferred task.

### ToolCorrectnessMetric
Assesses the agent's tool/function calling accuracy by comparing tools called against expected tools.

### StepEfficiencyMetric
Evaluates the efficiency of execution steps in completing the task.

```python
from src.evaluation import evaluate_with_deepeval, DEEPEVAL_AVAILABLE

# Evaluate agent response with Gemini
results = evaluate_with_deepeval(
    query="How many zombie instances?",
    response=agent_response,
    tools_used=["filter_underutilized_vms", "calculate_total_savings"],
    model_name="gemini-1.5-flash"  # Uses Gemini as judge
)

print(f"Task Completion: {results['results']['task_completion']['score']:.0%}")
print(f"Tool Correctness: {results['results']['tool_correctness']['score']:.0%}")
```

## Exa MCP Integration

Uses **Exa MCP Server** directly for live Azure VM pricing:

```python
from crewai.mcp import MCPServerHTTP

exa_mcp = MCPServerHTTP(
    url=f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}",
    headers={"Content-Type": "application/json"}
)

architect = Agent(
    role="Cloud Solutions Architect",
    mcps=[exa_mcp],  # Direct MCP integration
    ...
)
```

### Exa MCP Tools Available
- `web_search_exa` - Search for Azure VM pricing
- `crawling_exa` - Extract detailed pricing from URLs
- `company_research_exa` - Research cloud providers

## Setup

```bash
cd cloud-lease-negotiator
pip install -r requirements.txt
```

### API Keys Required

1. **Groq API Key** (FREE): https://console.groq.com/keys
2. **Exa API Key**: https://dashboard.exa.ai/
3. **Google/Gemini API Key** (for DeepEval): https://aistudio.google.com/apikey

Add to `.env`:
```
GROQ_API_KEY=your_groq_key
EXA_API_KEY=your_exa_key
GOOGLE_API_KEY=your_gemini_key
```

## Run

```bash
streamlit run app.py
```

## Data Format

### VM Usage Data (cloud_cluster_dataset.csv)
```csv
vm_id,current_size,cpu_cores,ram_gb,monthly_cost_usd,avg_cpu_usage_percent,avg_ram_usage_percent,cluster_id
vm-100000,Standard_B2s,2,4,22.11,26.72,14.23,cluster-1
```

## 3-Stage Pipeline

1. **Auditor** - Filters underutilized VMs (CPU < 30% AND RAM < 30%)
2. **Architect** - LLM Sense analysis + Exa MCP for live pricing
3. **CFO** - Calculates total savings

## Memory

Uses Mem0 with FREE HuggingFace embeddings - no OpenAI key needed.
