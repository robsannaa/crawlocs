# All Docs MCP

A smart documentation crawler and search system that automatically indexes documentation websites and makes them searchable through an MCP server.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <your-repo>
cd all-docs-mcp

# Install dependencies
cd docs_manager
uv sync
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=docs-mcp
OPENAI_API_KEY=your_openai_api_key
```

### 3. Crawl Documentation

```bash
cd docs_manager
python3 populate_db.py --library_name pandas --url https://pandas.pydata.org/docs/
```

### 4. Start the MCP Server

```bash
cd mcp_server
uv sync
python3 main.py
```

### 5. Use the Streamlit App

```bash
# In the root directory
streamlit run app.py
```

### 6. Test with MCP Inspector (Optional)

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is a powerful tool for testing and debugging MCP servers. Use it to:

- Test search functionality directly
- Debug server responses
- Explore available tools and resources
- Monitor server logs

```bash
# Test the MCP server directly
npx @modelcontextprotocol/inspector \
  uv \
  --directory mcp_server \
  run \
  mcp-server \
  --port 8000
```

**Inspector Features:**

- **Resources Tab**: Browse indexed documentation
- **Tools Tab**: Test search functionality
- **Prompts Tab**: Try different search queries
- **Notifications**: Monitor server logs and errors

## 🔄 System Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documentation │    │   Crawl4AI       │    │   Pinecone      │
│   Website       │───▶│   Deep Crawler   │───▶│   Vector DB     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   OpenAI         │    │   MCP Server    │
                       │   Embeddings     │    │   (Search API)  │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Streamlit     │
                                               │   UI            │
                                               └─────────────────┘
```

## 📁 Project Structure

```
all-docs-mcp/
├── docs_manager/          # Crawler and indexing service
│   ├── populate_db.py     # Main crawler script
│   ├── vector_store.py    # Pinecone integration
│   └── pyproject.toml     # Dependencies
├── mcp_server/           # MCP server for search
│   ├── main.py           # Server entry point
│   └── pyproject.toml    # Dependencies
├── app.py                # Streamlit UI
└── README.md            # This file
```

## 🎯 Features

- **Smart Crawling**: Uses crawl4ai to deeply crawl documentation sites
- **Content Filtering**: Automatically skips gallery/index pages
- **Vector Search**: Stores content in Pinecone for semantic search
- **MCP Integration**: Provides search through MCP protocol
- **Web UI**: Streamlit interface for testing and exploration

## 🔧 Configuration

### Crawler Settings

- **Max Depth**: 20 levels deep
- **Max Pages**: 10,000 pages per library
- **Word Threshold**: 50 words minimum
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters

### Supported Libraries

- pandas
- numpy
- matplotlib
- Any documentation site

## 🐛 Troubleshooting

**"No documents found"**: Check if the library name matches the namespace in Pinecone
**"Metadata size exceeds limit"**: Content is automatically truncated
**"Crawl stopped early"**: Check the URL and site accessibility

### MCP Inspector Issues

**"Server connection failed"**:

- Ensure the MCP server is running (`python3 main.py` in mcp_server/)
- Check if port 8000 is available
- Verify environment variables are set

**"No tools available"**:

- Make sure you've crawled some documentation first
- Check Pinecone index has data in the correct namespace
- Verify the server is properly initialized

**"Search returns no results"**:

- Test with a simple query like "documentation" or "help"
- Check the library name matches your crawled data
- Verify the Pinecone index contains the expected data

## 📝 Example Usage

```bash
# Crawl pandas documentation
python3 populate_db.py --library_name pandas --url https://pandas.pydata.org/docs/

# Crawl matplotlib documentation
python3 populate_db.py --library_name matplotlib --url https://matplotlib.org/stable/

# Search through the UI
streamlit run app.py
```

That's it! The system will crawl, index, and make your documentation searchable. 🎉
