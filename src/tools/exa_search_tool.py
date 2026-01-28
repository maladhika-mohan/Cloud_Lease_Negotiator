"""Exa MCP Search Tool - Uses Exa MCP Server for web search.

Communicates with exa-mcp-server via JSON-RPC over stdio.
"""
import os
import subprocess
import json
import threading
import queue
import platform
from crewai.tools import BaseTool
from pathlib import Path


class ExaMCPClient:
    """Client to communicate with Exa MCP Server via JSON-RPC over stdio."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to reuse MCP connection."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.process = None
        self.request_id = 0
        self.response_queue = queue.Queue()
        self.reader_thread = None
        self.tools_available = []
        self._initialized = True
        
    def start(self) -> bool:
        """Start the MCP server process."""
        if self.process and self.process.poll() is None:
            return True
            
        exa_key = os.getenv("EXA_API_KEY")
        if not exa_key:
            return False
        
        try:
            env = os.environ.copy()
            env["EXA_API_KEY"] = exa_key
            
            # Platform-specific command
            if platform.system() == "Windows":
                self.process = subprocess.Popen(
                    ["npx", "-y", "exa-mcp-server"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                    bufsize=0,
                    shell=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                self.process = subprocess.Popen(
                    ["npx", "-y", "exa-mcp-server"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                    bufsize=0
                )
            
            # Start reader thread
            self.reader_thread = threading.Thread(target=self._read_responses, daemon=True)
            self.reader_thread.start()
            
            # Initialize MCP connection
            init_response = self._send_and_wait("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cloud-lease-negotiator", "version": "1.0.0"}
            }, timeout=30)
            
            if not init_response:
                self._log_stderr()
                return False
            
            # Send initialized notification
            self._send_notification("notifications/initialized", {})
            
            # List available tools
            tools_response = self._send_and_wait("tools/list", {}, timeout=10)
            if tools_response and "tools" in tools_response.get("result", {}):
                self.tools_available = [t["name"] for t in tools_response["result"]["tools"]]
            
            return True
            
        except Exception as e:
            print(f"MCP start error: {e}")
            self._log_stderr()
            return False
    
    def _log_stderr(self):
        """Log any stderr output for debugging."""
        if self.process and self.process.stderr:
            try:
                import select
                if platform.system() != "Windows":
                    ready, _, _ = select.select([self.process.stderr], [], [], 0.1)
                    if ready:
                        err = self.process.stderr.read(1000)
                        if err:
                            print(f"MCP stderr: {err}")
            except Exception:
                pass
    
    def _read_responses(self):
        """Read responses from MCP server stdout."""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            response = json.loads(line)
                            self.response_queue.put(response)
                        except json.JSONDecodeError:
                            pass
            except Exception:
                break
    
    def _send_request(self, method: str, params: dict) -> int:
        """Send JSON-RPC request to MCP server."""
        if not self.process or self.process.poll() is not None:
            return -1
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            return self.request_id
        except Exception:
            return -1
    
    def _send_notification(self, method: str, params: dict):
        """Send JSON-RPC notification (no response expected)."""
        if not self.process or self.process.poll() is not None:
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        try:
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()
        except Exception:
            pass
    
    def _send_and_wait(self, method: str, params: dict, timeout: int = 30) -> dict:
        """Send request and wait for response."""
        # Clear queue first
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break
        
        req_id = self._send_request(method, params)
        if req_id < 0:
            return None
        
        try:
            response = self.response_queue.get(timeout=timeout)
            return response
        except queue.Empty:
            return None
    
    def call_tool(self, tool_name: str, arguments: dict, timeout: int = 45) -> dict:
        """Call an MCP tool and return the result."""
        if not self.process or self.process.poll() is not None:
            if not self.start():
                return {"error": "Failed to start MCP server"}
        
        response = self._send_and_wait("tools/call", {
            "name": tool_name,
            "arguments": arguments
        }, timeout=timeout)
        
        if not response:
            return {"error": "Timeout waiting for MCP response"}
        
        if "result" in response:
            return response["result"]
        elif "error" in response:
            return {"error": response["error"]}
        return response
    
    def get_available_tools(self) -> list:
        """Return list of available tool names."""
        return self.tools_available
    
    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.process = None


def get_mcp_client() -> ExaMCPClient:
    """Get singleton MCP client instance."""
    return ExaMCPClient()


class ExaSearchTool(BaseTool):
    name: str = "exa_web_search"
    description: str = """Search the web using Exa MCP Server.
    
    Use this tool to find:
    - Current Azure/AWS/GCP VM pricing
    - Latest cloud cost optimization strategies  
    - Market comparisons and alternatives
    - Technical documentation
    
    Input: search query string (e.g., "Azure VM Standard_D4s_v3 monthly pricing USD")
    
    Returns: Relevant web content with sources."""

    def _run(self, query: str) -> str:
        """Execute web search using Exa MCP Server."""
        if not os.getenv("EXA_API_KEY"):
            return "⚠️ EXA_API_KEY not set. Cannot perform web search."
        
        mcp = get_mcp_client()
        
        if not mcp.start():
            return "⚠️ Failed to start Exa MCP server. Check EXA_API_KEY and npx installation."
        
        # Check available tools
        tools = mcp.get_available_tools()
        
        # Try different tool names that Exa MCP might use
        tool_name = None
        for possible_name in ["web_search_exa", "search", "exa_search", "webSearch"]:
            if possible_name in tools:
                tool_name = possible_name
                break
        
        if not tool_name and tools:
            # Use first available tool if no match
            tool_name = tools[0]
        elif not tool_name:
            tool_name = "web_search_exa"  # Default fallback
        
        result = mcp.call_tool(tool_name, {
            "query": query,
            "numResults": 5
        })
        
        if "error" in result:
            return f"⚠️ MCP search error: {result['error']}"
        
        # Parse MCP response
        return self._format_response(query, result)
    
    def _format_response(self, query: str, result: dict) -> str:
        """Format MCP response into readable output."""
        output = f"## Web Search Results: {query}\n\n"
        
        # Handle different response formats
        content = None
        if isinstance(result, dict):
            content = result.get("content", [])
        elif isinstance(result, list):
            content = result
        
        if not content:
            output += str(result)[:2000]
            return output
        
        for i, item in enumerate(content[:5], 1):
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    output += f"### Result {i}\n{text[:800]}\n\n---\n\n"
                else:
                    output += f"### Result {i}\n{str(item)[:800]}\n\n---\n\n"
            else:
                output += f"### Result {i}\n{str(item)[:800]}\n\n---\n\n"
        
        return output


class ExaCrawlTool(BaseTool):
    name: str = "exa_crawl_url"
    description: str = """Crawl and extract content from a specific URL using Exa MCP.
    
    Input: URL to crawl
    Returns: Extracted text content from the page."""

    def _run(self, url: str) -> str:
        """Crawl a URL using Exa MCP Server."""
        if not os.getenv("EXA_API_KEY"):
            return "⚠️ EXA_API_KEY not set."
        
        mcp = get_mcp_client()
        
        if not mcp.start():
            return "⚠️ Failed to start Exa MCP server."
        
        tools = mcp.get_available_tools()
        
        # Try different crawl tool names
        tool_name = None
        for possible_name in ["crawling_exa", "crawl", "getContents", "extract"]:
            if possible_name in tools:
                tool_name = possible_name
                break
        
        if not tool_name:
            tool_name = "crawling_exa"
        
        result = mcp.call_tool(tool_name, {
            "url": url,
            "maxCharacters": 3000
        })
        
        if "error" in result:
            return f"⚠️ MCP crawl error: {result['error']}"
        
        # Parse response
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and content:
                return content[0].get("text", str(content))
        
        return str(result)[:3000]
