import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain.docstore.document import Document

load_dotenv()

# Get Pinecone configuration from environment variables
index_name = os.getenv("PINECONE_INDEX_NAME", "docs-mcp")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Initialize OpenAI embeddings with text-embedding-3-large model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=1536, request_timeout=30
)


def search_documents(library: str, query: str, k: int = 10) -> list[Document]:
    """Searches using the raw Pinecone client within the specified library namespace."""

    namespace = library.lower()
    index = pc.Index(index_name)

    if not query.strip():
        print("Empty query provided. Returning no results as a generic search is unreliable.")
        return []

    print(f"Searching for query: '{query.strip()}' in namespace: '{namespace}'")

    # 1. Create embedding for the query
    try:
        query_embedding = embeddings.embed_query(query.strip())
        print(f"Successfully generated query embedding.")
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

    # 2. Query Pinecone
    try:
        query_results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
        )
        print(f"Pinecone query returned {len(query_results.get('matches', []))} matches.")
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

    # 3. Construct Document objects
    results = []
    matches = query_results.get("matches", [])
    for match in matches:
        metadata = match.get("metadata", {})
        # The text is in the metadata, pop it to use for page_content
        page_content = metadata.pop("chunk_text", "")
        doc = Document(page_content=page_content, metadata=metadata)
        results.append(doc)

    print(f"Constructed {len(results)} Document objects from results.")
    return results


def search_docs_as_dicts(library: str, query: str, k: int = 5) -> list[dict]:
    """
    Searches for documents and returns them as a list of dictionaries.
    This is the centralized function to be used by the MCP server and Streamlit app.
    """
    print("\n--- [vector_store.search_docs_as_dicts] ---")
    print(f"Calling search_documents with: library='{library}', query='{query}', k={k}")
    results = search_documents(library, query, k)

    dict_results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in results
    ]
    print(f"Returning {len(dict_results)} results as dictionaries.")
    print("-----------------------------------------\n")
    return dict_results
