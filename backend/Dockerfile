# Backend Development Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy backend source code
COPY config.yaml ./config.yaml
COPY backend ./backend
COPY skills ./skills

# Install dependencies
RUN cd backend && uv sync

# Expose ports
EXPOSE 8001 2024

# Start both LangGraph server (background) and Gateway (foreground)
CMD ["sh", "-c", "cd /app/backend && uv run langgraph dev --no-browser --allow-blocking --host 0.0.0.0 --port 2024 & sleep 5 && cd /app/backend && uv run uvicorn src.gateway.app:app --host 0.0.0.0 --port 8001"]
