"""Main entry point for Cloud Lease Negotiator.

Strategy: Hybrid Python-Filtering & LLM Sense Chunking
- Stage 1: Hard filtering via Pandas (deterministic, reduces tokens 60-70%)
- Stage 2: LLM Sense reasoning with token-aware chunking (~50 VMs/batch)
- Stage 3: Python Code Interpreter for guaranteed accurate financial aggregation

LLM: Groq with Llama 3.3 70B (FAST inference, FREE tier)
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from src.crew import create_crew

def main():
    """Run the Cloud Lease Negotiator crew."""
    load_dotenv()
    
    # Ensure Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        print("Get your FREE key from: https://console.groq.com/keys")
        print("Set it in .env file or export GROQ_API_KEY=your-key")
        return
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CLOUD LEASE NEGOTIATOR - VM Right-Sizing Agent")
    print("Strategy: Hybrid Python-Filtering & LLM Sense Chunking")
    print("LLM: Groq - Llama 3.3 70B (FAST inference)")
    print("=" * 70)
    
    # Get user query
    print("\nExample queries:")
    print("  - Find all VMs that can be downsized")
    print("  - Identify wasteful resources in our cloud infrastructure")
    print("  - Which machines are over-provisioned?")
    
    user_query = input("\nEnter your query (or press Enter for default): ").strip()
    if not user_query:
        user_query = "Find all VMs that are underutilized and can be downsized to save costs"
    
    print(f"\nProcessing: {user_query}")
    print("=" * 70)
    
    # Create and run crew
    crew = create_crew(user_query)
    result = crew.kickoff()
    
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result)
    
    # Check if savings report was created
    report_path = output_dir / "savings_report.csv"
    if report_path.exists():
        print(f"\nDetailed recommendations saved to: {report_path}")

if __name__ == "__main__":
    main()
