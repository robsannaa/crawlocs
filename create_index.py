import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docs-mcp")

# Configuration for the desired OpenAI embedding model
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small
# --- End Configuration ---


def setup_pinecone_index():
    """
    Deletes the existing Pinecone index (if it exists) and creates a new,
    standard index configured for the specified embedding model dimensions.
    """
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables.")
        return

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # 1. Delete the old index to ensure a clean start
        if PINECONE_INDEX_NAME in [index.name for index in pc.list_indexes()]:
            print(f"Found existing index '{PINECONE_INDEX_NAME}'. Deleting it...")
            pc.delete_index(PINECONE_INDEX_NAME)
            print("Index deleted. Waiting 15 seconds for the system to update...")
            time.sleep(15)
        else:
            print(f"Index '{PINECONE_INDEX_NAME}' not found. Proceeding to creation.")

        # 2. Create a new standard index with the correct dimension
        print(f"Creating new standard index '{PINECONE_INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}...")
        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )

        print("\n✅ Success! Index creation command sent to Pinecone.")
        print("Please wait a few moments for the index to become ready in the Pinecone console.")
        print("After it's ready, you can run your application to crawl and search.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("If the error mentions the index already exists, please delete it manually from the Pinecone console and run this script again.")

if __name__ == "__main__":
    setup_pinecone_index()