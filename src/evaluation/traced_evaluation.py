"""Traced Evaluation Module for Cloud Lease Negotiator.

Uses DeepEval's TaskCompletion and ToolCorrectness metrics with Gemini as judge.
"""
import os
from typing import Dict, List, Any

try:
    from deepeval.metrics import (
        TaskCompletionMetric,
        ToolCorrectnessMetric,
    )
    from deepeval.models import GeminiModel
    from deepeval.test_case import LLMTestCase, ToolCall
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


class TracedAgentEvaluator:
    """Evaluator with Gemini for DeepEval metrics."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.gemini_model = None
        self.results_history = []
        self._init_model()
    
    def _init_model(self):
        """Initialize Gemini model."""
        if not DEEPEVAL_AVAILABLE:
            return
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            self.gemini_model = GeminiModel(
                model_name=self.model_name,
                api_key=api_key,
                temperature=0
            )
    
    def evaluate(
        self,
        query: str,
        response: str,
        tools_called: List[str],
        expected_tools: List[str] = None
    ) -> Dict[str, Any]:
        """Run evaluation on agent response."""
        if not DEEPEVAL_AVAILABLE:
            return {"error": "DeepEval not available"}
        
        if not self.gemini_model:
            self._init_model()
            if not self.gemini_model:
                return {"error": "Gemini API key required"}
        
        # Expected = actual for deterministic pipeline
        if expected_tools is None:
            expected_tools = tools_called
        
        tools_called_objs = [ToolCall(name=t) for t in tools_called]
        expected_tools_objs = [ToolCall(name=t) for t in expected_tools]
        
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            tools_called=tools_called_objs,
            expected_tools=expected_tools_objs
        )
        
        results = {
            "query": query,
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "model": self.model_name,
            "metrics": {}
        }
        
        scores = []
        
        # Task Completion
        try:
            metric = TaskCompletionMetric(
                threshold=self.threshold,
                model=self.gemini_model,
                include_reason=True,
                async_mode=False
            )
            metric.measure(test_case)
            results["metrics"]["task_completion"] = {
                "score": metric.score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', '')
            }
            scores.append(metric.score)
        except Exception as e:
            results["metrics"]["task_completion"] = {"error": str(e), "score": 0}
            scores.append(0)
        
        # Tool Correctness
        try:
            metric = ToolCorrectnessMetric(
                threshold=self.threshold,
                include_reason=True,
                should_exact_match=False,
                should_consider_ordering=False
            )
            metric.measure(test_case)
            results["metrics"]["tool_correctness"] = {
                "score": metric.score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', '')
            }
            scores.append(metric.score)
        except Exception as e:
            results["metrics"]["tool_correctness"] = {"error": str(e), "score": 0}
            scores.append(0)
        
        results["overall_score"] = sum(scores) / len(scores) if scores else 0
        results["passed"] = results["overall_score"] >= self.threshold
        
        self.results_history.append(results)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        if not self.results_history:
            return {"message": "No evaluations yet"}
        
        total = len(self.results_history)
        passed = sum(1 for r in self.results_history if r.get("passed", False))
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0
        }


def create_evaluator(model_name: str = "gemini-2.5-flash-lite", threshold: float = 0.5):
    """Create a TracedAgentEvaluator instance."""
    return TracedAgentEvaluator(model_name=model_name, threshold=threshold)
