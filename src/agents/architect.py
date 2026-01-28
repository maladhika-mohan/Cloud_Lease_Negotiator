"""Architect Agent - LLM Sense with batch processing."""
from crewai import Agent
from src.tools.savings_tool import SavingsLoggerTool, ClearReportTool
from src.tools.llm_sense_tool import LLMSenseTool
from src.tools.batch_tool import BatchAnalyzeTool
from src.config import get_llm


def create_architect_agent() -> Agent:
    return Agent(
        role="Cloud Solutions Architect",
        goal="Analyze VMs semantically and log right-sizing recommendations",
        backstory="""You apply LLM Sense - converting raw metrics into semantic meaning.
Use analyze_vm_semantically for detailed analysis, or batch_analyze_and_log for quick processing.
Log recommendations using log_recommendation with format: vm_id,current_size,cost,new_size,new_cost""",
        tools=[LLMSenseTool(), SavingsLoggerTool(), ClearReportTool(), BatchAnalyzeTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=12
    )
