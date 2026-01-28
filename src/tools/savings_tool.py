"""Stage 3: Savings Calculator Tools - Python-based financial accuracy.

Uses deterministic Python/Pandas calculations instead of LLM arithmetic.
This prevents hallucinations and ensures 100% accuracy on financial math.
"""
from crewai.tools import BaseTool
import pandas as pd
import csv
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Built-in pricing for rightsizing calculations
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
    
    return (best_match, best_cost) if best_match else ("Standard_B1s", 7.59)


class SavingsCalculatorTool(BaseTool):
    name: str = "calculate_total_savings"
    description: str = """Calculate total savings using Python/Pandas - NOT LLM arithmetic.
    
    This tool uses DETERMINISTIC code for 100% accurate financial calculations.
    It computes SUM(Old_Price - New_Price) programmatically.
    
    Commands:
    - 'summary' - Full financial summary with totals
    - 'calculate_all' - Calculate rightsizing savings for ALL zombie VMs
    - 'top_N' (e.g., 'top_5') - Show top N highest-impact recommendations
    
    Returns accurate financial report with monthly and annual savings."""

    def _run(self, command: str) -> str:
        """Calculate savings using Python - deterministic, no LLM math."""
        try:
            cmd = command.strip().lower()
            
            if cmd == 'calculate_all':
                return self._calculate_all_savings()
            
            report_path = OUTPUT_DIR / "savings_report.csv"
            
            if not report_path.exists():
                # Auto-calculate if no report exists
                return self._calculate_all_savings()
            
            df = pd.read_csv(report_path)
            
            if len(df) == 0:
                return self._calculate_all_savings()
            
            if cmd == 'summary':
                return self._financial_summary(df)
            
            elif cmd.startswith('top_'):
                n = int(cmd.split('_')[1])
                return self._top_recommendations(df, n)
            
            else:
                return "Commands: 'summary', 'calculate_all', 'top_N'"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _calculate_all_savings(self) -> str:
        """Calculate rightsizing savings for ALL zombie VMs using Python."""
        DATA_DIR = OUTPUT_DIR.parent
        df = pd.read_csv(DATA_DIR / "cloud_cluster_dataset.csv")
        
        # Filter ALL underutilized VMs
        waste_mask = (df['avg_cpu_usage_percent'] < 30) & (df['avg_ram_usage_percent'] < 30)
        filtered = df[waste_mask].sort_values('monthly_cost_usd', ascending=False)
        
        report_path = OUTPUT_DIR / "savings_report.csv"
        if report_path.exists():
            report_path.unlink()
        
        results = []
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vm_id', 'current_size', 'current_cost', 
                           'recommended_size', 'new_cost', 'monthly_savings'])
            
            for _, vm in filtered.iterrows():
                # Calculate effective usage with 50% buffer
                eff_cpu = vm['cpu_cores'] * (vm['avg_cpu_usage_percent'] / 100)
                eff_ram = vm['ram_gb'] * (vm['avg_ram_usage_percent'] / 100)
                min_cpu = max(1, eff_cpu * 1.5)
                min_ram = max(1, eff_ram * 1.5)
                
                rec_instance, rec_cost = find_best_instance(min_cpu, min_ram)
                
                if rec_cost < vm['monthly_cost_usd']:
                    savings = vm['monthly_cost_usd'] - rec_cost
                    writer.writerow([
                        vm['vm_id'], vm['current_size'], vm['monthly_cost_usd'],
                        rec_instance, rec_cost, savings
                    ])
                    results.append({
                        'current': vm['monthly_cost_usd'],
                        'new': rec_cost,
                        'savings': savings
                    })
        
        # Calculate totals using Python (NOT LLM)
        total_current = sum(r['current'] for r in results)
        total_new = sum(r['new'] for r in results)
        total_savings = sum(r['savings'] for r in results)
        annual_savings = total_savings * 12
        savings_pct = (total_savings / total_current * 100) if total_current > 0 else 0
        
        output = "\n## ROI CALCULATION (Python-Computed, 100% Accurate)\n\n"
        output += "**Method:** `SUM(Old_Price - New_Price)` computed programmatically\n\n"
        output += "| Metric | Value |\n"
        output += "|--------|-------|\n"
        output += f"| VMs Analyzed | {len(results):,} |\n"
        output += f"| Current Monthly Cost | ${total_current:,.2f} |\n"
        output += f"| Projected Monthly Cost | ${total_new:,.2f} |\n"
        output += f"| **Monthly Savings** | **${total_savings:,.2f}** |\n"
        output += f"| **Annual Savings** | **${annual_savings:,.2f}** |\n"
        output += f"| Cost Reduction | {savings_pct:.1f}% |\n"
        
        return output
    
    def _financial_summary(self, df: pd.DataFrame) -> str:
        """Generate financial summary from saved report."""
        total_current = df['current_cost'].sum()
        total_new = df['new_cost'].sum()
        total_savings = df['monthly_savings'].sum()
        annual_savings = total_savings * 12
        savings_pct = (total_savings / total_current * 100) if total_current > 0 else 0
        
        output = "\n## FINANCIAL SUMMARY (Python-Calculated)\n\n"
        output += "| Metric | Value |\n"
        output += "|--------|-------|\n"
        output += f"| VMs Analyzed | {len(df):,} |\n"
        output += f"| Current Monthly Cost | ${total_current:,.2f} |\n"
        output += f"| Projected Monthly Cost | ${total_new:,.2f} |\n"
        output += f"| **Monthly Savings** | **${total_savings:,.2f}** |\n"
        output += f"| **Annual Savings** | **${annual_savings:,.2f}** |\n"
        output += f"| Cost Reduction | {savings_pct:.1f}% |\n"
        
        return output
    
    def _top_recommendations(self, df: pd.DataFrame, n: int) -> str:
        """Show top N highest-impact recommendations."""
        top = df.nlargest(n, 'monthly_savings')
        
        output = f"\n## TOP {n} HIGHEST-IMPACT RECOMMENDATIONS\n\n"
        output += "| VM ID | Current | Recommended | Monthly Savings |\n"
        output += "|-------|---------|-------------|----------------|\n"
        for _, row in top.iterrows():
            output += f"| {row['vm_id']} | {row['current_size']} | {row['recommended_size']} | ${row['monthly_savings']:,.2f} |\n"
        
        return output


class SavingsLoggerTool(BaseTool):
    name: str = "log_recommendation"
    description: str = """Log a single VM recommendation. Format: 'vm_id,current_size,current_cost,recommended_size,new_cost'"""

    def _run(self, recommendation: str) -> str:
        try:
            report_path = OUTPUT_DIR / "savings_report.csv"
            parts = [p.strip() for p in recommendation.split(",")]
            if len(parts) != 5:
                return "Error: Need 5 values: vm_id,current_size,current_cost,recommended_size,new_cost"
            
            vm_id, current_size, current_cost, recommended_size, new_cost = parts
            current_cost = float(current_cost)
            new_cost = float(new_cost)
            savings = current_cost - new_cost
            
            file_exists = report_path.exists()
            with open(report_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['vm_id', 'current_size', 'current_cost', 
                                   'recommended_size', 'new_cost', 'monthly_savings'])
                writer.writerow([vm_id, current_size, current_cost, 
                               recommended_size, new_cost, savings])
            
            return f"Logged: {vm_id} saves ${savings:,.2f}/month"
        except Exception as e:
            return f"Error: {str(e)}"


class ClearReportTool(BaseTool):
    name: str = "clear_savings_report"
    description: str = """Clear the savings report to start fresh."""

    def _run(self, _: str = "") -> str:
        try:
            report_path = OUTPUT_DIR / "savings_report.csv"
            if report_path.exists():
                report_path.unlink()
            return "Savings report cleared."
        except Exception as e:
            return f"Error: {str(e)}"
