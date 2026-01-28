"""Pricing Search Tool - Uses Exa MCP directly for live Azure VM pricing.

Exa MCP Server: https://mcp.exa.ai/mcp
Documentation: https://github.com/exa-labs/exa-mcp-server

This tool integrates with Exa's hosted MCP server for real-time web search.
"""
from crewai.tools import BaseTool


class PricingSearchTool(BaseTool):
    name: str = "search_vm_pricing"
    description: str = """Search for Azure VM pricing using Exa MCP.
    
    This tool is a placeholder - actual pricing search is done via Exa MCP
    integrated directly into the Architect agent using mcps=[exa_mcp].
    
    The Architect agent can use Exa's web_search_exa tool with queries like:
    "Azure VM Standard_D4s_v3 monthly pricing USD pay-as-you-go"
    
    Input: VM instance type (e.g., 'Standard_D4s_v3')
    """

    def _run(self, instance_type: str) -> str:
        """Return guidance for using Exa MCP."""
        return f"""To get live pricing for {instance_type}:

Use Exa MCP web_search_exa with query:
"Azure VM {instance_type} monthly pricing USD pay-as-you-go 2024"

The Architect agent has Exa MCP integrated via mcps=[exa_mcp] parameter.
This enables direct access to Exa's web search tools for live pricing data.

Sources to search:
- cloudprice.net/vm/{instance_type}
- instances.vantage.sh/azure/vm/{instance_type.lower().replace('_', '-')}
- azure.microsoft.com/pricing/details/virtual-machines/
"""


class RightsizingTool(BaseTool):
    name: str = "get_rightsizing_recommendation"
    description: str = """Get VM rightsizing recommendation based on usage.
    
    Input format: 'cpu_cores,ram_gb,avg_cpu_percent,avg_ram_percent'
    Example: '8,32,15,20' (8 cores, 32GB RAM, 15% CPU usage, 20% RAM usage)
    
    Returns recommended smaller instance type based on effective usage + 50% buffer."""

    def _run(self, input_str: str) -> str:
        """Calculate rightsizing recommendation."""
        try:
            parts = [float(x.strip()) for x in input_str.split(',')]
            if len(parts) != 4:
                return "Error: Need 4 values: cpu_cores,ram_gb,avg_cpu_percent,avg_ram_percent"
            
            cpu_cores, ram_gb, avg_cpu, avg_ram = parts
            
            # Calculate effective usage with 50% buffer
            eff_cpu = cpu_cores * (avg_cpu / 100)
            eff_ram = ram_gb * (avg_ram / 100)
            min_cpu = max(1, eff_cpu * 1.5)
            min_ram = max(1, eff_ram * 1.5)
            
            # Recommend instance family based on workload pattern
            if avg_ram > avg_cpu * 1.5:
                family = "E-series (memory-optimized)"
            elif avg_cpu > avg_ram * 1.5:
                family = "F-series (compute-optimized)"
            else:
                family = "D-series (general purpose)"
            
            return f"""## Rightsizing Recommendation

**Current Usage:**
- Effective CPU: {eff_cpu:.1f} cores (of {cpu_cores})
- Effective RAM: {eff_ram:.1f} GB (of {ram_gb})

**Minimum Required (with 50% buffer):**
- CPU: {min_cpu:.1f} cores
- RAM: {min_ram:.1f} GB

**Recommended Family:** {family}

**Next Step:** Use Exa MCP to search for current pricing:
"Azure VM {family.split()[0]} {int(min_cpu)} vCPU {int(min_ram)}GB monthly pricing"
"""
            
        except Exception as e:
            return f"Error: {str(e)}"
