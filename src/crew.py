"""CrewAI Crew for Cloud Cost Optimization with MCP Integration.

All chat queries go through the 3-agent pipeline.
Uses Python/Pandas for accurate calculations.
Integrates Exa API for web search when needed.
"""
from crewai import Agent, Crew, Task
from src.tools import WasteFilterTool, SavingsCalculatorTool, LLMSenseTool
from src.tools.batch_tool import BatchAnalyzeTool
from src.tools.exa_search_tool import ExaSearchTool
from src.config import get_llm


def create_chat_crew(user_query: str, use_web_search: bool = False) -> Crew:
    """Create 3-agent crew for any chat query about cloud infrastructure."""
    
    llm = get_llm()
    q = user_query.lower()
    
    # Base tools for all agents
    auditor_tools = [WasteFilterTool()]
    architect_tools = [LLMSenseTool(), BatchAnalyzeTool()]
    cfo_tools = [SavingsCalculatorTool()]
    
    # Add web search tool if needed
    if use_web_search:
        exa_tool = ExaSearchTool()
        architect_tools.append(exa_tool)
    
    # Agent 1: Auditor
    auditor = Agent(
        role="Cloud Infrastructure Auditor",
        goal="Discover and filter underutilized VMs using Python/Pandas tools",
        backstory="You are an expert at finding wasteful cloud resources. Always use filter_underutilized_vms tool. Commands: zombie, near_zero, premium, cluster_analysis, top_5",
        tools=auditor_tools,
        llm=llm,
        verbose=True,
        max_iter=3
    )
    
    # Agent 2: Architect
    architect = Agent(
        role="Cloud Solutions Architect", 
        goal="Analyze VMs and recommend rightsizing with detailed reasoning",
        backstory="You provide detailed technical analysis. Use analyze_vm_semantically for VM deep-dives. Use batch_analyze_and_log to process all VMs. Use exa_web_search for current pricing if available.",
        tools=architect_tools,
        llm=llm,
        verbose=True,
        max_iter=3
    )
    
    # Agent 3: CFO
    cfo = Agent(
        role="Cloud Finance Officer",
        goal="Provide accurate financial summary using Python calculations",
        backstory="You are the financial expert. ALWAYS use calculate_total_savings tool. Present results conversationally with exact numbers formatted as $X,XXX.XX",
        tools=cfo_tools,
        llm=llm,
        verbose=True,
        max_iter=2
    )
    
    # Create tasks based on query type
    tasks = _create_tasks_for_query(q, user_query, auditor, architect, cfo)
    
    return Crew(
        agents=[auditor, architect, cfo],
        tasks=tasks,
        verbose=True
    )


def _create_tasks_for_query(q, user_query, auditor, architect, cfo):
    """Create appropriate tasks based on query type."""
    
    if any(kw in q for kw in ['calculate', 'exact', 'total', 'savings', 'roi', 'financial', 'downsize all']):
        return _roi_tasks(user_query, auditor, architect, cfo)
    elif any(kw in q for kw in ['example', 'recommend', 'specific', 'pick', 'show me', 'top']):
        return _example_tasks(user_query, auditor, architect, cfo)
    elif any(kw in q for kw in ['cluster', 'distribution', 'worst offender', 'which cluster']):
        return _cluster_tasks(user_query, auditor, architect, cfo)
    elif any(kw in q for kw in ['premium', 'm-series', 'l-series', 'expensive']):
        return _premium_tasks(user_query, auditor, architect, cfo)
    elif any(kw in q for kw in ['pricing', 'price', 'search', 'web', 'current', 'market', 'aws', 'gcp']):
        return _pricing_tasks(user_query, auditor, architect, cfo)
    else:
        return _discovery_tasks(user_query, auditor, architect, cfo)


def _roi_tasks(user_query, auditor, architect, cfo):
    audit_task = Task(
        description="Use filter_underutilized_vms with 'zombie'. Report the EXACT zombie count and wasted cost from the tool output.",
        expected_output="Zombie count (e.g., 3,251) and total wasted monthly cost",
        agent=auditor
    )
    architect_task = Task(
        description="Use batch_analyze_and_log with 'run' to calculate rightsizing for ALL VMs.",
        expected_output="Batch processing results with total savings",
        agent=architect,
        context=[audit_task]
    )
    cfo_task = Task(
        description=f"User asked: {user_query}\n\nIMPORTANT: Use the AUDITOR's zombie count from filter_underutilized_vms (NOT VMs Analyzed from savings). Present: zombie count from auditor, current wasted cost from auditor, projected savings, annual savings, percentage reduction.",
        expected_output="Financial report using auditor's zombie count",
        agent=cfo,
        context=[audit_task, architect_task]
    )
    return [audit_task, architect_task, cfo_task]


def _example_tasks(user_query, auditor, architect, cfo):
    audit_task = Task(
        description=f"User asked: {user_query}\n\nUse filter_underutilized_vms 'top_5' to get highest-cost zombies.",
        expected_output="Top 5 VM IDs with costs",
        agent=auditor
    )
    architect_task = Task(
        description="For 3 VMs from auditor's list: use analyze_vm_semantically, explain why wasteful, recommend new type, show savings. Then use batch_analyze_and_log 'run'.",
        expected_output="3 detailed VM analyses plus total savings",
        agent=architect,
        context=[audit_task]
    )
    cfo_task = Task(
        description=f"User asked: {user_query}\n\nUse calculate_total_savings 'summary'. Present example VMs, total savings, next steps.",
        expected_output="Summary with examples and total savings",
        agent=cfo,
        context=[architect_task]
    )
    return [audit_task, architect_task, cfo_task]


def _cluster_tasks(user_query, auditor, architect, cfo):
    audit_task = Task(
        description=f"User asked: {user_query}\n\nUse filter_underutilized_vms 'cluster_analysis' to find worst clusters.",
        expected_output="Cluster analysis with worst offender",
        agent=auditor
    )
    architect_task = Task(
        description="Explain why certain clusters have more zombies, which to prioritize, remediation approach.",
        expected_output="Cluster insights and remediation suggestions",
        agent=architect,
        context=[audit_task]
    )
    cfo_task = Task(
        description=f"User asked: {user_query}\n\nUse calculate_total_savings 'summary'. Present worst cluster cost, top 3 cluster savings, ROI.",
        expected_output="Cluster-focused financial analysis",
        agent=cfo,
        context=[architect_task]
    )
    return [audit_task, architect_task, cfo_task]


def _premium_tasks(user_query, auditor, architect, cfo):
    audit_task = Task(
        description=f"User asked: {user_query}\n\nUse filter_underutilized_vms 'premium' to find M-series and L-series waste.",
        expected_output="Premium instance waste report",
        agent=auditor
    )
    architect_task = Task(
        description="Explain why M/L-series are expensive, suggest downgrades, identify termination candidates.",
        expected_output="Premium rightsizing recommendations",
        agent=architect,
        context=[audit_task]
    )
    cfo_task = Task(
        description=f"User asked: {user_query}\n\nUse calculate_total_savings 'summary'. Present premium waste, savings potential, why high-impact.",
        expected_output="Premium savings analysis",
        agent=cfo,
        context=[architect_task]
    )
    return [audit_task, architect_task, cfo_task]


def _discovery_tasks(user_query, auditor, architect, cfo):
    audit_task = Task(
        description=f"User asked: {user_query}\n\nUse filter_underutilized_vms 'zombie' to find all zombie instances. Report the EXACT zombie count and wasted cost from the tool output.",
        expected_output="Zombie count (e.g., 3,251) and total wasted monthly cost from filter tool",
        agent=auditor
    )
    architect_task = Task(
        description="Use batch_analyze_and_log 'run' to calculate rightsizing savings. Note: VMs processed may be less than total zombies (some can't be rightsized cheaper).",
        expected_output="Batch processing results with savings amount",
        agent=architect,
        context=[audit_task]
    )
    cfo_task = Task(
        description=f"User asked: {user_query}\n\nIMPORTANT: Use the AUDITOR's zombie count (from filter_underutilized_vms), NOT the VMs Analyzed count from savings. Present: 1) Zombie count from auditor, 2) Wasted cost from auditor, 3) Potential savings from calculate_total_savings, 4) Top recommendations.",
        expected_output="Report using auditor's zombie count and savings calculations",
        agent=cfo,
        context=[audit_task, architect_task]
    )
    return [audit_task, architect_task, cfo_task]


def _pricing_tasks(user_query, auditor, architect, cfo):
    """Tasks for pricing/web search queries using Exa MCP."""
    audit_task = Task(
        description=f"User asked: {user_query}\n\nUse filter_underutilized_vms 'top_5' to identify VMs that need pricing comparison.",
        expected_output="Top VMs for pricing analysis",
        agent=auditor
    )
    architect_task = Task(
        description=f"User asked: {user_query}\n\nUse exa_web_search tool to search for current cloud VM pricing. Search query: 'Azure VM pricing USD 2024' or similar based on user query. Then compare with our VM costs.",
        expected_output="Web search results with pricing data",
        agent=architect,
        context=[audit_task]
    )
    cfo_task = Task(
        description=f"User asked: {user_query}\n\nUse calculate_total_savings 'summary'. Present pricing findings from web search, compare with our costs, provide recommendations.",
        expected_output="Pricing analysis with market comparison",
        agent=cfo,
        context=[architect_task]
    )
    return [audit_task, architect_task, cfo_task]


def create_crew(user_query: str = "Analyze VMs") -> Crew:
    """Backward compatible function."""
    return create_chat_crew(user_query, use_web_search=False)
