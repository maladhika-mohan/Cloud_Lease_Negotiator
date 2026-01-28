"""Stage 1: Hard Filtering Tool - Pandas-based deterministic filtering.

Handles Big Data by using Python/Pandas instead of LLM reading raw rows.
This prevents context window crashes and ensures accuracy.
"""
from crewai.tools import BaseTool
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent


class WasteFilterTool(BaseTool):
    name: str = "filter_underutilized_vms"
    description: str = """Filter the VM dataset to find underutilized (wasteful) resources using Python/Pandas.
    
    This tool uses DETERMINISTIC code filtering - NOT LLM reading raw data.
    This prevents context window crashes on large datasets.
    
    Commands:
    - 'all' - Get summary of ALL underutilized VMs (CPU < 30% AND RAM < 30%)
    - 'zombie' - Same as 'all', finds "zombie" instances
    - 'near_zero' - Find VMs with CPU < 10% AND RAM < 10% (critical waste)
    - 'premium' or 'm_series' or 'l_series' - Find underutilized premium instances
    - 'cluster_analysis' - Analyze waste distribution by cluster
    - 'top_N' (e.g., 'top_5') - Get top N highest-cost underutilized VMs
    
    Returns filtered statistics and sample VMs for further analysis."""

    def _run(self, command: str) -> str:
        """Filter dataset using Pandas - deterministic, no LLM reasoning on raw data."""
        try:
            df = pd.read_csv(DATA_DIR / "cloud_cluster_dataset.csv")
            cmd = command.strip().lower()
            
            # Base filter: underutilized VMs
            waste_mask = (df['avg_cpu_usage_percent'] < 30) & (df['avg_ram_usage_percent'] < 30)
            filtered = df[waste_mask].copy()
            filtered = filtered.sort_values('monthly_cost_usd', ascending=False)
            
            if cmd in ['all', 'zombie']:
                return self._summary_report(df, filtered)
            
            elif cmd == 'near_zero':
                near_zero = df[(df['avg_cpu_usage_percent'] < 10) & (df['avg_ram_usage_percent'] < 10)]
                return self._near_zero_report(df, near_zero)
            
            elif cmd in ['premium', 'm_series', 'l_series']:
                return self._premium_report(df, filtered)
            
            elif cmd == 'cluster_analysis':
                return self._cluster_report(filtered)
            
            elif cmd.startswith('top_'):
                n = int(cmd.split('_')[1])
                return self._top_n_report(filtered, n)
            
            else:
                return "Commands: 'all', 'zombie', 'near_zero', 'premium', 'cluster_analysis', 'top_N'"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _summary_report(self, df: pd.DataFrame, filtered: pd.DataFrame) -> str:
        """Generate summary of all underutilized VMs."""
        total_cost = filtered['monthly_cost_usd'].sum()
        top_vms = filtered.head(5)
        
        output = "\n## ZOMBIE INSTANCE DISCOVERY (Python/Pandas Filtered)\n\n"
        output += "**Filter Applied:** `df.loc[(cpu < 30) & (ram < 30)]`\n\n"
        output += "| Metric | Value |\n"
        output += "|--------|-------|\n"
        output += f"| Total VMs in Dataset | {len(df):,} |\n"
        output += f"| Zombie Instances Found | {len(filtered):,} |\n"
        output += f"| Monthly Cost (Wasted) | ${total_cost:,.2f} |\n"
        output += f"| Percentage of Fleet | {(len(filtered)/len(df)*100):.1f}% |\n\n"
        
        output += "### Top 5 Highest-Cost Zombies (for detailed analysis)\n\n"
        output += "| VM ID | Instance Type | Monthly Cost | CPU % | RAM % |\n"
        output += "|-------|---------------|--------------|-------|-------|\n"
        for _, vm in top_vms.iterrows():
            output += f"| {vm['vm_id']} | {vm['current_size']} | ${vm['monthly_cost_usd']:,.2f} | {vm['avg_cpu_usage_percent']:.1f}% | {vm['avg_ram_usage_percent']:.1f}% |\n"
        
        return output
    
    def _near_zero_report(self, df: pd.DataFrame, near_zero: pd.DataFrame) -> str:
        """Report on near-zero utilization VMs."""
        total_cost = near_zero['monthly_cost_usd'].sum()
        
        output = "\n## NEAR-ZERO UTILIZATION VMs (Critical Waste)\n\n"
        output += "**Filter Applied:** `df.loc[(cpu < 10) & (ram < 10)]`\n\n"
        output += "| Metric | Value |\n"
        output += "|--------|-------|\n"
        output += f"| Near-Zero VMs Found | {len(near_zero):,} |\n"
        output += f"| Monthly Cost (Wasted) | ${total_cost:,.2f} |\n"
        output += f"| **Recommendation** | Consider termination |\n\n"
        
        if len(near_zero) > 0:
            top = near_zero.sort_values('monthly_cost_usd', ascending=False).head(5)
            output += "### Top 5 Near-Zero VMs\n\n"
            output += "| VM ID | Instance Type | Monthly Cost | CPU % | RAM % |\n"
            output += "|-------|---------------|--------------|-------|-------|\n"
            for _, vm in top.iterrows():
                output += f"| {vm['vm_id']} | {vm['current_size']} | ${vm['monthly_cost_usd']:,.2f} | {vm['avg_cpu_usage_percent']:.1f}% | {vm['avg_ram_usage_percent']:.1f}% |\n"
        
        return output
    
    def _premium_report(self, df: pd.DataFrame, filtered: pd.DataFrame) -> str:
        """Report on premium M-series and L-series waste."""
        # M-series filter
        m_series = df[df['current_size'].str.contains('_M', case=False, na=False)]
        m_waste = m_series[(m_series['avg_cpu_usage_percent'] < 30) & (m_series['avg_ram_usage_percent'] < 30)]
        m_near_zero = m_series[(m_series['avg_cpu_usage_percent'] < 10) & (m_series['avg_ram_usage_percent'] < 10)]
        
        # L-series filter
        l_series = df[df['current_size'].str.contains('_L', case=False, na=False)]
        l_waste = l_series[(l_series['avg_cpu_usage_percent'] < 30) & (l_series['avg_ram_usage_percent'] < 30)]
        l_near_zero = l_series[(l_series['avg_cpu_usage_percent'] < 10) & (l_series['avg_ram_usage_percent'] < 10)]
        
        output = "\n## PREMIUM INSTANCE WASTE ANALYSIS\n\n"
        output += "**Business Logic:** Targeting high-cost M-series and L-series instances\n\n"
        
        output += "### M-Series (Memory Optimized - Premium)\n\n"
        output += "| Metric | Value |\n"
        output += "|--------|-------|\n"
        output += f"| Total M-Series | {len(m_series):,} |\n"
        output += f"| Underutilized (< 30%) | {len(m_waste):,} |\n"
        output += f"| Near-Zero (< 10%) | {len(m_near_zero):,} |\n"
        output += f"| Wasted Monthly Cost | ${m_waste['monthly_cost_usd'].sum():,.2f} |\n\n"
        
        if len(m_near_zero) > 0:
            output += "**⚠️ ALERT: Premium M-Series at Near-Zero Utilization:**\n\n"
            output += "| VM ID | Instance Type | Monthly Cost | CPU % | RAM % |\n"
            output += "|-------|---------------|--------------|-------|-------|\n"
            for _, vm in m_near_zero.head(5).iterrows():
                output += f"| {vm['vm_id']} | {vm['current_size']} | ${vm['monthly_cost_usd']:,.2f} | {vm['avg_cpu_usage_percent']:.1f}% | {vm['avg_ram_usage_percent']:.1f}% |\n"
            output += "\n"
        
        output += "### L-Series (Storage Optimized - Premium)\n\n"
        output += "| Metric | Value |\n"
        output += "|--------|-------|\n"
        output += f"| Total L-Series | {len(l_series):,} |\n"
        output += f"| Underutilized (< 30%) | {len(l_waste):,} |\n"
        output += f"| Near-Zero (< 10%) | {len(l_near_zero):,} |\n"
        output += f"| Wasted Monthly Cost | ${l_waste['monthly_cost_usd'].sum():,.2f} |\n"
        
        return output
    
    def _cluster_report(self, filtered: pd.DataFrame) -> str:
        """Analyze waste distribution by cluster."""
        cluster_stats = filtered.groupby('cluster_id').agg({
            'vm_id': 'count',
            'monthly_cost_usd': 'sum'
        }).rename(columns={'vm_id': 'zombie_count', 'monthly_cost_usd': 'wasted_cost'})
        
        cluster_stats = cluster_stats.sort_values('zombie_count', ascending=False)
        worst = cluster_stats.head(10)
        
        output = "\n## CLUSTER ANALYSIS (Pattern Recognition)\n\n"
        output += "**Analysis:** Using `groupby('cluster_id')` to identify waste hotspots\n\n"
        
        top_cluster = cluster_stats.index[0]
        top_count = cluster_stats.iloc[0]['zombie_count']
        top_cost = cluster_stats.iloc[0]['wasted_cost']
        
        output += f"### 🚨 Worst Offender: {top_cluster}\n\n"
        output += f"- **Zombie VMs:** {int(top_count):,}\n"
        output += f"- **Wasted Monthly Cost:** ${top_cost:,.2f}\n\n"
        
        output += "### Top 10 Clusters by Zombie Count\n\n"
        output += "| Cluster ID | Zombie VMs | Wasted Cost |\n"
        output += "|------------|------------|-------------|\n"
        for cluster_id, row in worst.iterrows():
            output += f"| {cluster_id} | {int(row['zombie_count']):,} | ${row['wasted_cost']:,.2f} |\n"
        
        return output
    
    def _top_n_report(self, filtered: pd.DataFrame, n: int) -> str:
        """Get top N highest-cost underutilized VMs."""
        top = filtered.head(n)
        
        output = f"\n## TOP {n} HIGHEST-COST ZOMBIE VMs\n\n"
        output += "| VM ID | Instance Type | Monthly Cost | CPU % | RAM % | Cluster |\n"
        output += "|-------|---------------|--------------|-------|-------|--------|\n"
        for _, vm in top.iterrows():
            output += f"| {vm['vm_id']} | {vm['current_size']} | ${vm['monthly_cost_usd']:,.2f} | {vm['avg_cpu_usage_percent']:.1f}% | {vm['avg_ram_usage_percent']:.1f}% | {vm['cluster_id']} |\n"
        
        return output
