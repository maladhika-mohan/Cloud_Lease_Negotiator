"""LLM Sense Tool - Semantic interpretation of VM metrics.

This tool provides "LLM Sense" - converting raw metrics into semantic context.
It interprets technical specifications and provides human-like reasoning.
For live pricing, it can trigger Exa MCP web search.
"""
from crewai.tools import BaseTool
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent

# Built-in Azure pricing reference (fallback when Exa MCP not available)
AZURE_PRICING = {
    "Standard_B1s": {"cpu": 1, "ram": 1, "cost": 7.59, "family": "Burstable"},
    "Standard_B2s": {"cpu": 2, "ram": 4, "cost": 30.37, "family": "Burstable"},
    "Standard_B4ms": {"cpu": 4, "ram": 16, "cost": 60.74, "family": "Burstable"},
    "Standard_D2s_v3": {"cpu": 2, "ram": 8, "cost": 70.08, "family": "General Purpose"},
    "Standard_D4s_v3": {"cpu": 4, "ram": 16, "cost": 140.16, "family": "General Purpose"},
    "Standard_D8s_v3": {"cpu": 8, "ram": 32, "cost": 280.32, "family": "General Purpose"},
    "Standard_D16s_v3": {"cpu": 16, "ram": 64, "cost": 560.64, "family": "General Purpose"},
    "Standard_E2s_v3": {"cpu": 2, "ram": 16, "cost": 91.98, "family": "Memory Optimized"},
    "Standard_E4s_v3": {"cpu": 4, "ram": 32, "cost": 183.96, "family": "Memory Optimized"},
    "Standard_E8s_v3": {"cpu": 8, "ram": 64, "cost": 367.92, "family": "Memory Optimized"},
    "Standard_F2s_v2": {"cpu": 2, "ram": 4, "cost": 61.32, "family": "Compute Optimized"},
    "Standard_F4s_v2": {"cpu": 4, "ram": 8, "cost": 122.64, "family": "Compute Optimized"},
    "Standard_F8s_v2": {"cpu": 8, "ram": 16, "cost": 245.28, "family": "Compute Optimized"},
}

# Instance family classification
INSTANCE_FAMILIES = {
    "_D": "General Purpose (balanced CPU/RAM)",
    "_E": "Memory Optimized (high RAM)",
    "_F": "Compute Optimized (high CPU)",
    "_M": "Memory Optimized Premium (very high RAM)",
    "_L": "Storage Optimized (high disk I/O)",
    "_B": "Burstable (variable workloads)",
    "_N": "GPU Enabled (ML/AI workloads)",
}


def get_instance_family(instance_type: str) -> str:
    """Identify the Azure instance family from the type name."""
    for pattern, family in INSTANCE_FAMILIES.items():
        if pattern in instance_type:
            return family
    return "Unknown Family"


def find_best_instance(min_cpu: float, min_ram: float) -> tuple:
    """Find cheapest instance that meets requirements."""
    best_match = None
    best_cost = float('inf')
    
    for instance, specs in AZURE_PRICING.items():
        if specs["cpu"] >= min_cpu and specs["ram"] >= min_ram:
            if specs["cost"] < best_cost:
                best_cost = specs["cost"]
                best_match = instance
    
    return (best_match, AZURE_PRICING.get(best_match, {})) if best_match else ("Standard_B1s", AZURE_PRICING["Standard_B1s"])


class LLMSenseTool(BaseTool):
    name: str = "analyze_vm_semantically"
    description: str = """Analyze a VM using LLM Sense - semantic interpretation of metrics.
    
    This tool provides QUALITATIVE analysis, not just numbers:
    - Interprets instance family (e.g., "D-series = General Purpose")
    - Classifies workload pattern (idle, light, moderate, heavy)
    - Cross-references pricing catalog for rightsizing
    - Provides human-like reasoning for recommendations
    
    Input: vm_id (e.g., 'vm-100011')
    
    For live pricing from web, use Exa MCP separately with query:
    "Azure VM [instance_type] monthly pricing USD"
    """

    def _run(self, vm_id: str) -> str:
        """Provide semantic analysis with LLM Sense."""
        try:
            df = pd.read_csv(DATA_DIR / "cloud_cluster_dataset.csv")
            vm = df[df['vm_id'] == vm_id.strip()]
            
            if len(vm) == 0:
                return f"VM {vm_id} not found"
            
            vm = vm.iloc[0]
            return self._semantic_analysis(vm)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _semantic_analysis(self, vm) -> str:
        """Generate semantic analysis with reasoning."""
        # Classify workload intensity
        cpu_pct = vm['avg_cpu_usage_percent']
        ram_pct = vm['avg_ram_usage_percent']
        
        cpu_level = "idle" if cpu_pct < 10 else "light" if cpu_pct < 30 else "moderate" if cpu_pct < 60 else "heavy"
        ram_level = "idle" if ram_pct < 10 else "light" if ram_pct < 30 else "moderate" if ram_pct < 60 else "heavy"
        
        # Identify workload pattern
        if cpu_pct > ram_pct * 2:
            workload_pattern = "Compute-bound (batch processing, calculations)"
        elif ram_pct > cpu_pct * 2:
            workload_pattern = "Memory-bound (caching, in-memory database)"
        else:
            workload_pattern = "Balanced (web server, general application)"
        
        # Waste assessment
        if cpu_pct < 10 and ram_pct < 10:
            waste_level = "🔴 CRITICAL"
            waste_reason = "Near-zero utilization - candidate for termination"
        elif cpu_pct < 30 and ram_pct < 30:
            waste_level = "🟠 HIGH"
            waste_reason = "Both CPU and RAM severely underutilized"
        else:
            waste_level = "🟡 MODERATE"
            waste_reason = "Partial underutilization"
        
        # Instance family interpretation
        current_family = get_instance_family(vm['current_size'])
        
        # Calculate rightsizing
        effective_cpu = vm['cpu_cores'] * (cpu_pct / 100)
        effective_ram = vm['ram_gb'] * (ram_pct / 100)
        min_cpu = max(1, effective_cpu * 1.5)
        min_ram = max(1, effective_ram * 1.5)
        
        rec_instance, rec_specs = find_best_instance(min_cpu, min_ram)
        rec_cost = rec_specs.get('cost', 7.59)
        rec_family = rec_specs.get('family', 'Burstable')
        
        savings = vm['monthly_cost_usd'] - rec_cost
        savings_pct = (savings / vm['monthly_cost_usd']) * 100
        
        output = f"""
## Semantic Analysis: {vm['vm_id']}

### Current State
| Attribute | Value | Interpretation |
|-----------|-------|----------------|
| Instance Type | {vm['current_size']} | {current_family} |
| vCPUs | {vm['cpu_cores']} | {cpu_level} usage ({cpu_pct:.1f}%) |
| RAM | {vm['ram_gb']} GB | {ram_level} usage ({ram_pct:.1f}%) |
| Monthly Cost | ${vm['monthly_cost_usd']:,.2f} | |
| Cluster | {vm['cluster_id']} | |

### LLM Sense Interpretation

**Workload Pattern:** {workload_pattern}

**Waste Assessment:** {waste_level}
- {waste_reason}
- Effective CPU used: {effective_cpu:.2f} cores (of {vm['cpu_cores']})
- Effective RAM used: {effective_ram:.2f} GB (of {vm['ram_gb']} GB)

### Rightsizing Recommendation

| Metric | Current | Recommended |
|--------|---------|-------------|
| Instance | {vm['current_size']} | {rec_instance} |
| Family | {current_family} | {rec_family} |
| Monthly Cost | ${vm['monthly_cost_usd']:,.2f} | ${rec_cost:,.2f} |
| **Savings** | - | **${savings:,.2f}/month ({savings_pct:.0f}%)** |

### Reasoning

This VM ({vm['current_size']}) belongs to the **{current_family}** family but is only using 
{cpu_pct:.1f}% CPU and {ram_pct:.1f}% RAM. The workload pattern suggests {workload_pattern.lower()}.

With a 50% safety buffer, the actual workload only requires {min_cpu:.1f} vCPUs and {min_ram:.1f} GB RAM.
The recommended **{rec_instance}** ({rec_family}) provides adequate resources at ${rec_cost:,.2f}/month,
resulting in **${savings:,.2f} monthly savings** ({savings_pct:.0f}% reduction).

*For live pricing verification, use Exa MCP: "Azure VM {rec_instance} monthly pricing USD"*
"""
        return output
