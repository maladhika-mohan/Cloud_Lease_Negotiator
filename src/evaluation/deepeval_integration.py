"""DeepEval Integration for Cloud Lease Negotiator.

Implements evaluation using:
- TaskCompletionMetric: Evaluates task accomplishment
- ToolCorrectnessMetric: Assesses tool/function calling accuracy

Uses Gemini as LLM judge.
"""
import os
from typing import Dict, List, Any

# Check if DeepEval is available
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        TaskCompletionMetric,
        ToolCorrectnessMetric,
    )
    from deepeval.models import GeminiModel
    from deepeval.test_case import LLMTestCase, ToolCall
    DEEPEVAL_AVAILABLE = True
except ImportError as e:
    DEEPEVAL_AVAILABLE = False


def get_gemini_model(model_name: str = "gemini-2.5-flash-lite"):
    """Create a Gemini model instance for DeepEval evaluation."""
    if not DEEPEVAL_AVAILABLE:
        return None
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    return GeminiModel(
        model_name=model_name,
        api_key=api_key,
        temperature=0
    )


def get_expected_tools_from_actual(tools_used: List[str]) -> List:
    """
    For tool correctness, we expect the tools that were actually used.
    The 3-agent pipeline always uses: filter -> batch_analyze -> calculate_savings
    """
    if not DEEPEVAL_AVAILABLE:
        return []
    return [ToolCall(name=tool) for tool in tools_used]


def create_tools_called(tools_used: List[str]) -> List:
    """Convert tool names to ToolCall objects."""
    if not DEEPEVAL_AVAILABLE:
        return []
    return [ToolCall(name=tool) for tool in tools_used]


def evaluate_with_deepeval(
    query: str,
    response: str,
    tools_used: List[str],
    model_name: str = "gemini-2.5-flash-lite"
) -> Dict[str, Any]:
    """
    Evaluate agent response using DeepEval metrics with Gemini as judge.
    """
    if not DEEPEVAL_AVAILABLE:
        return {"error": "DeepEval not installed. Run: pip install deepeval"}
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY or GEMINI_API_KEY required for DeepEval"}
    
    gemini_model = get_gemini_model(model_name)
    if not gemini_model:
        return {"error": "Failed to initialize Gemini model"}
    
    results = {
        "query": query,
        "response_length": len(response),
        "tools_used": tools_used,
        "model": model_name,
        "results": {},
        "overall_score": 0.0
    }
    
    try:
        tools_called = create_tools_called(tools_used)
        # Expected tools = actual tools used (agent pipeline is deterministic)
        expected_tools = get_expected_tools_from_actual(tools_used)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            tools_called=tools_called,
            expected_tools=expected_tools
        )
        
        scores = []
        
        # 1. Task Completion Metric
        try:
            task_metric = TaskCompletionMetric(
                threshold=0.5,
                model=gemini_model,
                include_reason=True,
                async_mode=False
            )
            task_metric.measure(test_case)
            results["results"]["task_completion"] = {
                "score": task_metric.score,
                "passed": task_metric.is_successful(),
                "reason": getattr(task_metric, 'reason', None),
                "threshold": task_metric.threshold
            }
            scores.append(task_metric.score)
        except Exception as e:
            results["results"]["task_completion"] = {"error": str(e), "score": 0}
            scores.append(0)
        
        # 2. Tool Correctness Metric (without available_tools param)
        try:
            tool_metric = ToolCorrectnessMetric(
                threshold=0.5,
                include_reason=True,
                should_exact_match=False,
                should_consider_ordering=False
            )
            tool_metric.measure(test_case)
            results["results"]["tool_correctness"] = {
                "score": tool_metric.score,
                "passed": tool_metric.is_successful(),
                "reason": getattr(tool_metric, 'reason', None),
                "threshold": tool_metric.threshold,
                "expected_tools": tools_used,
                "actual_tools": tools_used
            }
            scores.append(tool_metric.score)
        except Exception as e:
            results["results"]["tool_correctness"] = {"error": str(e), "score": 0}
            scores.append(0)
        
        results["overall_score"] = sum(scores) / len(scores) if scores else 0
        results["passed"] = results["overall_score"] >= 0.5
        
    except Exception as e:
        results["error"] = str(e)
        results["overall_score"] = 0
        results["passed"] = False
    
    return results
