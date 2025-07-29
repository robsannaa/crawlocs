from fastmcp import FastMCP
import sys
import os
from pathlib import Path
import traceback
from dotenv import load_dotenv


from docs_manager.vector_store import search_docs_as_dicts

# Load environment variables from a .env file in the project root
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)


# 1. Create the MCP server instance
mcp_server = FastMCP(
    "AllDocs",
    "A server for retrieving documentation for any library.",
)


@mcp_server.tool(
    name="search_docs",
    description="Search for the most up-to-date documentation, guides, tutorials, and API references for any library or framework. Use this tool whenever you need current information about libraries, frameworks, tools, or technologies. This searches through comprehensive documentation that has been crawled and indexed, providing you with the latest available information rather than relying on potentially outdated training data.",
)
async def search_docs(library: str, query: str = "", k: int = 5) -> list[dict]:
    """
    Search for the most current documentation and guides for a specific library or framework.

    Args:
        library: The name of the library, framework, or technology (e.g., "react", "fastapi", "langchain")
        query: Your search query (e.g., "authentication", "how to use hooks")
        k: Number of most relevant results to return (default: 5)

    Returns:
        A list of dictionaries, where each dictionary represents a document with its metadata.
    """
    print(
        f"🔍 MCP Server: search_docs called with library='{library}', query='{query}', k={k}",
        file=sys.stderr,
    )
    try:
        # Use the centralized search function
        return search_docs_as_dicts(library, query, k)
    except Exception as e:
        print(f"❌ MCP Server: Exception in search_docs: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []


if __name__ == "__main__":
    import os

    # Check if we're running in Fly.io (HTTP mode) or locally (stdio mode)
    if os.getenv("FLY_APP_NAME"):
        # Running on Fly.io - use HTTP transport
        import uvicorn
        from fastapi import FastAPI

        # Create FastAPI app wrapper
        app = FastAPI(title="AllDocs MCP Server")

        # Add health check endpoint
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "alldocs-mcp"}

        # Run with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        # Running locally - use stdio transport for MCP clients
        mcp_server.run()
