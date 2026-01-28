"""Auditor Agent - Stage 1: Deterministic Filtering."""
from crewai import Agent
from src.tools.filter_tool import WasteFilterTool
from src.config import get_llm


def create_auditor_agent() -> Agent:
    return Agent(
        role="Cloud Resource Auditor",
        goal="Filter underutilized VMs using deterministic Python filtering",
        backstory="""You use Python/Pandas for deterministic filtering - no LLM reasoning needed.
Waste criteria: CPU < 30% AND RAM < 30%. Use filter_underutilized_vms 'all' to get stats.""",
        tools=[WasteFilterTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=2
    )
