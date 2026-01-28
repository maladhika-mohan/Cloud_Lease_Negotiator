"""Batch Processing Tool - Analyzes ALL underutilized VMs using built-in pricing."""
from crewai.tools import BaseTool
import pandas as pd
import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = DATA_DIR / "output"

# Built-in Azure pricing reference (no external catalog needed)
AZURE_PRICING = {
    "Standard_B1s": {"cpu": 1, "ram": 1, "cost": 7.59},
    "Standard_B2s": {"cpu": 2, "ram": 4, "cost": 30.37},
    "Standard_B4ms": {"cpu": 4, "ram": 16, "cost": 60.74},
    "Standard_D2s_v3": {"cpu": 2, "ram": 8, "cost": 70.08},
    "Standard_D4s_v3": {"cpu": 4, "ram": 16, "cost": 140.16},
    "Standard_D8s_v3": {"cpu": 8, "ram": 32, "cost": 280.32},
    "Standard_D16s_v3": {"cpu": 16, "ram": 64, "cost": 560.64},
    "Standard_E2s_v3": {"cpu": 2, "ram": 16, "cost": 91.98},
    "Standard_E4s_v3": {"cpu": 4, "ram": 32, "cost": 183.96},
    "Standard_E8s_v3": {"cpu": 8, "ram": 64, "cost": 367.92},
    "Standard_F2s_v2": {"cpu": 2, "ram": 4, "cost": 61.32},
    "Standard_F4s_v2": {"cpu": 4, "ram": 8, "cost": 122.64},
    "Standard_F8s_v2": {"cpu": 8, "ram": 16, "cost": 245.28},
}


def find_best_instance(min_cpu: float, min_ram: float) -> tuple:
    """Find cheapest instance that meets requirements."""
    best_match = None
    best_cost = float('inf')
    
    for instance, specs in AZURE_PRICING.items():
        if specs["cpu"] >= min_cpu and specs["ram"] >= min_ram:
            if specs["cost"] < best_cost:
                best_cost = specs["cost"]
                best_match = instance
    
    if best_match:
        return best_match, AZURE_PRICING[best_match]["cost"]
    return None, None


class BatchAnalyzeTool(BaseTool):
    name: str = "batch_analyze_and_log"
    description: str = """Analyze ALL underutilized VMs and log recommendations.
    
    Input: 'run' to process ALL wasteful VMs
    
    Uses built-in Azure pricing reference (no external catalog needed).
    For live pricing, use Exa MCP web_search_exa separately.
    
    This tool:
    1. Filters ALL underutilized VMs (CPU < 30% AND RAM < 30%)
    2. Calculates rightsizing with 50% buffer
    3. Logs ALL recommendations to savings_report.csv"""

    def _run(self, command: str = "run") -> str:
        """Process ALL VMs using built-in pricing."""
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            report_path = OUTPUT_DIR / "savings_report.csv"
            
            if report_path.exists():
                report_path.unlink()
            
            df = pd.read_csv(DATA_DIR / "cloud_cluster_dataset.csv")
            
            # Filter ALL underutilized VMs
            waste_mask = (df['avg_cpu_usage_percent'] < 30) & (df['avg_ram_usage_percent'] < 30)
            filtered = df[waste_mask].sort_values('monthly_cost_usd', ascending=False)
            
            total_underutilized = len(filtered)
            
            results = []
            with open(report_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['vm_id', 'current_size', 'current_cost', 
                               'recommended_size', 'new_cost', 'monthly_savings'])
                
                for _, vm in filtered.iterrows():
                    eff_cpu = vm['cpu_cores'] * (vm['avg_cpu_usage_percent'] / 100)
                    eff_ram = vm['ram_gb'] * (vm['avg_ram_usage_percent'] / 100)
                    
                    min_cpu = max(1, eff_cpu * 1.5)
                    min_ram = max(1, eff_ram * 1.5)
                    
                    rec_instance, rec_cost = find_best_instance(min_cpu, min_ram)
                    
                    if rec_instance and rec_cost < vm['monthly_cost_usd']:
                        savings = vm['monthly_cost_usd'] - rec_cost
                        writer.writerow([
                            vm['vm_id'], vm['current_size'], vm['monthly_cost_usd'],
                            rec_instance, rec_cost, savings
                        ])
                        results.append(savings)
            
            total_savings = sum(results)
            return f"\n## Batch Analysis Complete\n\n| Metric | Value |\n|--------|-------|\n| VMs Processed | {len(results):,} |\n| Total Underutilized | {total_underutilized:,} |\n| **Monthly Savings** | **${total_savings:,.2f}** |\n"
            
        except Exception as e:
            return f"Error: {str(e)}"
