# Use Python 3.11 slim image
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy pyproject.toml and uv.lock for dependency management
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the MCP server code
COPY main.py .

# Create docs_manager directory and copy vector_store.py
RUN mkdir -p docs_manager
COPY vector_store.py ./docs_manager/

# Expose port (MCP servers typically use stdio, but Fly.io needs a port)
EXPOSE 8080

# Run the MCP server using uv
CMD ["uv", "run", "python", "main.py"] 