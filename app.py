import streamlit as st
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Page config
st.set_page_config(page_title="AllDocs MCP Manager", layout="wide")

# Add docs_manager to path for imports
sys.path.append(str(Path(__file__).parent / "docs_manager"))

# Constants
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docs-mcp")
LIBRARY_PLACEHOLDER = "e.g., nhost, langchain, fastapi"
URL_PLACEHOLDER = "e.g., https://docs.nhost.io/welcome"


def get_pinecone_status() -> Tuple[List[str], Optional[dict]]:
    """Get Pinecone connection status and available namespaces."""
    try:
        from pinecone import Pinecone
        from dotenv import load_dotenv

        load_dotenv()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
        return namespaces, stats
    except Exception as e:
        st.error(f"❌ Failed to connect to Pinecone: {e}")
        return [], None


def test_mcp_search(
    library: str, query: str = "", k: int = 5
) -> Tuple[List[dict], Optional[str]]:
    """Test the exact same search function used by MCP server."""
    try:
        from docs_manager.vector_store import search_docs_as_dicts

        results = search_docs_as_dicts(library, query, k)
        print(f"Results: {results}")
        return results, None
    except Exception as e:
        import traceback

        return [], f"Error: {str(e)}\n\n{traceback.format_exc()}"


def display_search_results(results: List[dict], error: Optional[str]):
    """Display search results or error."""
    if error:
        st.error(f"❌ Search failed:")
        st.code(error, language="python")
    elif not results:
        st.warning("⚠️ No results found")
        st.info("This matches what the MCP server would return")
    else:
        st.success(f"✅ Found {len(results)} results")
        print(results)

        for i, result in enumerate(results, 1):
            page_content = result.get("page_content", "")
            metadata = result.get("metadata", {})
            title = metadata.get("title", "No Title")
            source = metadata.get("source", "No Source URL")

            with st.expander(f"Result {i}: {title}"):
                st.markdown(f"**Source:** [{source}]({source})")
                st.markdown("---")
                st.text(page_content)


def display_library_documents(library_name: str, k: int = 10):
    """Display documents from a specific library."""
    st.subheader(f"📚 Documents in '{library_name}'")

    # Get documents from this library
    results, error = test_mcp_search(library_name, "", k)

    if error:
        st.error(f"❌ Failed to load documents: {error}")
        return

    if not results:
        st.info(f"No documents found in library '{library_name}'")
        return

    st.success(f"📖 Found {len(results)} documents")

    # Add search within library
    with st.expander("🔍 Search within this library", expanded=True):
        search_within = st.text_input(
            "Search query",
            placeholder="e.g., authentication, setup, configuration",
            key=f"search_within_{library_name}",
            help="Search for specific content within this library",
        )

        if search_within:
            search_results, search_error = test_mcp_search(
                library_name, search_within, k
            )
            if search_error:
                st.error(f"Search failed: {search_error}")
            elif not search_results:
                st.warning("No results found for your search")
            else:
                st.success(f"Found {len(search_results)} matching documents")
                for i, result in enumerate(search_results, 1):
                    display_document_card(result, i, library_name)
        else:
            # Show all documents
            for i, result in enumerate(results, 1):
                display_document_card(result, i, library_name)


def display_document_card(result: dict, index: int, library_name: str):
    """Displays a single document with its full content inside an expander."""
    page_content = result.get("page_content", "")
    metadata = result.get("metadata", {})
    title = metadata.get("title", "No Title")
    source = metadata.get("source", "No Source URL")

    # Use an expander to show the full content directly
    with st.expander(f"**{index}. {title}**"):
        st.markdown(f"**Source:** [{source}]({source})")
        st.markdown("---")
        st.text(page_content)


def create_library_explorer(namespaces: List[str], stats: dict):
    """Create an interactive library explorer with document previews."""
    if not namespaces:
        st.info("No libraries found in Pinecone")
        return

    st.subheader("📚 Library Explorer")

    # Library selection
    selected_library = st.selectbox(
        "Choose a library to explore:",
        options=namespaces,
        format_func=lambda x: f"{x} ({stats.namespaces[x].vector_count} docs)",
        help="Select a library to view its documents",
    )

    if selected_library:
        # Number of documents to show
        k_docs = st.slider(
            "Number of documents to show:",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="How many documents to display from this library",
        )

        # Display documents for selected library
        display_library_documents(selected_library, k_docs)

        # Show library stats
        with st.expander("📊 Library Statistics"):
            ns_stats = stats.namespaces[selected_library]
            st.metric("Total Documents", ns_stats.vector_count)
            st.metric("Library Name", selected_library)


def run_crawler(library_name: str, start_url: str):
    """Run the crawler process with live output."""
    st.header("Crawling Progress")

    cmd = ["uv", "run", "python", "populate_db.py", library_name, start_url]

    st.info(f"🚀 Starting crawl for '{library_name}' at {start_url}...")
    st.info("This may take several minutes depending on the size of the documentation.")

    # Create container for live logs
    log_container = st.container()
    log_placeholder = log_container.empty()

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent / "docs_manager",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Collect live output
        live_output = []
        start_time = time.time()

        with log_placeholder:
            st.info("🔄 Starting crawler process...")

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                live_output.append(output.strip())

                # Update display every few lines or every 2 seconds
                if len(live_output) % 5 == 0 or (time.time() - start_time) > 2:
                    with log_placeholder:
                        st.code("\n".join(live_output[-50:]), language="bash")
                        st.caption(
                            f"📊 Total lines: {len(live_output)} | ⏱️ Running for {int(time.time() - start_time)}s"
                        )

        # Handle completion
        return_code = process.poll()

        if return_code == 0:
            st.success("✅ Crawling and upload completed successfully!")
        else:
            st.error("❌ Crawling failed!")

        # Show final output
        with log_placeholder:
            st.code("\n".join(live_output), language="bash")
            status = (
                "✅ Completed successfully"
                if return_code == 0
                else f"❌ Failed with code {return_code}"
            )
            st.caption(f"📊 Total lines: {len(live_output)} | {status}")

    except subprocess.TimeoutExpired:
        st.error("❌ Crawling timed out after 1 hour")
        st.info(
            "The documentation site might be very large. Check the logs for progress."
        )
    except Exception as e:
        st.error(f"❌ Error running crawler: {str(e)}")


def display_recent_logs():
    """Display recent crawl logs."""
    st.header("Recent Crawl Logs")

    docs_manager_path = Path(__file__).parent / "docs_manager"
    log_files = list(docs_manager_path.glob("crawl_*.log"))

    if log_files:
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for log_file in log_files[:3]:  # Show last 3 log files
            library = log_file.stem.replace("crawl_", "")

            with st.expander(f"📋 Logs for {library}"):
                try:
                    with open(log_file, "r") as f:
                        log_content = f.read()
                    st.code(log_content, language="bash")
                except Exception as e:
                    st.error(f"Could not read log file: {e}")
    else:
        st.info("No log files found. Start crawling to see logs here.")


def main():
    """Main application function."""
    st.title("AllDocs MCP Manager")

    # Get Pinecone status
    namespaces, stats = get_pinecone_status()

    if namespaces:
        st.success(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
        st.info(f"📚 Available libraries: {', '.join(namespaces)}")
    else:
        st.warning(
            f"⚠️ Connected to Pinecone index '{PINECONE_INDEX_NAME}' but no libraries found"
        )

    # Create 2-column layout
    col1, col2 = st.columns([1, 1])

    # LEFT COLUMN - MCP Search Tester
    with col1:
        st.header("🔍 MCP Search Tester")
        st.write("Test the exact same search function as the MCP server")

        # Search form
        with st.form("search_form"):
            library_name = st.text_input(
                "Library Name",
                placeholder=LIBRARY_PLACEHOLDER,
                help="The name of the library/framework (exactly like MCP server)",
            )

            search_query = st.text_input(
                "Search Query",
                placeholder="e.g., auth, hooks, setup",
                help="What you want to search for (optional - leave empty for random results)",
            )

            k_results = st.number_input(
                "Results (k)", min_value=1, max_value=20, value=5
            )
            search_button = st.form_submit_button("🔍 Test Search")

        # Handle search results
        if search_button:
            if not library_name:
                st.error("❌ Please enter a library name")
            else:
                st.header("Search Results")

                # Test the search
                search_description = f"Searching '{library_name}'"
                if search_query:
                    search_description += f" for '{search_query}'"
                search_description += "..."

                with st.spinner(search_description):
                    results, error = test_mcp_search(
                        library_name, search_query, k_results
                    )

                display_search_results(results, error)

        # Debug info
        st.header("Debug Info")
        st.code(
            f"""
MCP Server Function: search_docs(library: str, query: str = "", k: int = 5)
Search Method: Semantic search with embeddings (if query provided) or namespace filtering
Available Namespaces: {namespaces}
""",
            language="python",
        )

    # RIGHT COLUMN - Library Explorer & Upload
    with col2:
        st.header("📚 Library Explorer & Upload")

        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(
            ["🔍 Explore Libraries", "🚀 Upload Docs", "📋 Recent Logs"]
        )

        with tab1:
            st.write("Browse and search through your uploaded documentation")
            if namespaces and stats:
                create_library_explorer(namespaces, stats)
            else:
                st.info("No libraries found. Upload some documentation first!")

        with tab2:
            st.write("Crawl documentation sites and upload to Pinecone")

            # Upload form
            with st.form("upload_form"):
                library_name = st.text_input(
                    "Library Name",
                    placeholder=LIBRARY_PLACEHOLDER,
                    help="The name of the library/framework (will be used as Pinecone namespace)",
                )

                start_url = st.text_input(
                    "Documentation URL",
                    placeholder=URL_PLACEHOLDER,
                    help="The starting URL for the documentation site",
                )

                upload_button = st.form_submit_button("🚀 Start Crawling & Upload")

            # Handle upload
            if upload_button:
                if not library_name or not start_url:
                    st.error(
                        "❌ Please provide both library name and documentation URL"
                    )
                else:
                    run_crawler(library_name, start_url)

        with tab3:
            display_recent_logs()


if __name__ == "__main__":
    main()
