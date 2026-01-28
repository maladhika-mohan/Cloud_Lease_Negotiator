"""CFO Agent - Stage 3: Financial Aggregation."""
from crewai import Agent
from src.tools.savings_tool import SavingsCalculatorTool
from src.config import get_llm


def create_cfo_agent() -> Agent:
    return Agent(
        role="Cloud Finance Officer",
        goal="Calculate total savings using Python-based arithmetic",
        backstory="""You use Python/Pandas for guaranteed accurate calculations.
Use calculate_total_savings 'summary' to get financial totals from the savings report.""",
        tools=[SavingsCalculatorTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=2
    )
