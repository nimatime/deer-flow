import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from src.config.app_config import get_app_config
from src.gateway.config import get_gateway_config
from src.gateway.routers import (
    agents,
    artifacts,
    channels,
    mcp,
    memory,
    models,
    skills,
    suggestions,
    uploads,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

LANGGRAPH_URL = "http://localhost:2024"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    try:
        get_app_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        error_msg = f"Failed to load configuration during gateway startup: {e}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e
    config = get_gateway_config()
    logger.info(f"Starting API Gateway on {config.host}:{config.port}")

    try:
        from src.channels.service import start_channel_service
        channel_service = await start_channel_service()
        logger.info("Channel service started: %s", channel_service.get_status())
    except Exception:
        logger.exception("No IM channels configured or channel service failed to start")

    yield

    try:
        from src.channels.service import stop_channel_service
        await stop_channel_service()
    except Exception:
        logger.exception("Failed to stop channel service")
    logger.info("Shutting down API Gateway")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="DeerFlow API Gateway",
        description="""
## DeerFlow API Gateway

API Gateway for DeerFlow - A LangGraph-based AI agent backend with sandbox execution capabilities.

### Features

- **Models Management**: Query and retrieve available AI models
- **MCP Configuration**: Manage Model Context Protocol (MCP) server configurations
- **Memory Management**: Access and manage global memory data for personalized conversations
- **Skills Management**: Query and manage skills and their enabled status
- **Artifacts**: Access thread artifacts and generated files
- **Health Monitoring**: System health check endpoints

### Architecture

LangGraph requests are proxied to the LangGraph server running on port 2024.
This gateway provides custom endpoints for models, MCP configuration, skills, and artifacts.
        """,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "models", "description": "Operations for querying available AI models and their configurations"},
            {"name": "mcp", "description": "Manage Model Context Protocol (MCP) server configurations"},
            {"name": "memory", "description": "Access and manage global memory data for personalized conversations"},
            {"name": "skills", "description": "Manage skills and their configurations"},
            {"name": "artifacts", "description": "Access and download thread artifacts and generated files"},
            {"name": "uploads", "description": "Upload and manage user files for threads"},
            {"name": "agents", "description": "Create and manage custom agents with per-agent config and prompts"},
            {"name": "suggestions", "description": "Generate follow-up question suggestions for conversations"},
            {"name": "channels", "description": "Manage IM channel integrations (Feishu, Slack, Telegram)"},
            {"name": "health", "description": "Health check and system status endpoints"},
        ],
    )

    # CORS middleware for Railway deployment (no nginx)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(models.router)
    app.include_router(mcp.router)
    app.include_router(memory.router)
    app.include_router(skills.router)
    app.include_router(artifacts.router)
    app.include_router(uploads.router)
    app.include_router(agents.router)
    app.include_router(suggestions.router)
    app.include_router(channels.router)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "service": "deer-flow-gateway"}

    # ---- LangGraph Proxy Routes ----
    @app.api_route("/runs", methods=["GET", "POST", "PUT", "DELETE"], tags=["langgraph"])
    @app.api_route("/runs/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], tags=["langgraph"])
    async def proxy_runs(request: Request, path: str = ""):
        return await _proxy_to_langgraph(request)

    @app.api_route("/threads", methods=["GET", "POST", "PUT", "DELETE"], tags=["langgraph"])
    @app.api_route("/threads/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], tags=["langgraph"])
    async def proxy_threads(request: Request, path: str = ""):
        return await _proxy_to_langgraph(request)

    async def _proxy_to_langgraph(request: Request):
        """Proxy request to LangGraph server on localhost:2024."""
        url = f"{LANGGRAPH_URL}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "transfer-encoding", "connection")
        }
        body = await request.body()
        is_stream = "/stream" in request.url.path

        if is_stream:
            client = httpx.AsyncClient(timeout=None)
            try:
                req = client.build_request(
                    method=request.method, url=url, headers=headers, content=body
                )
                resp = await client.send(req, stream=True)

                async def stream_response():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()
                        await client.aclose()

                return StreamingResponse(
                    stream_response(),
                    status_code=resp.status_code,
                    headers={
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "connection")
                    },
                    media_type=resp.headers.get("content-type", "text/event-stream"),
                )
            except Exception:
                await client.aclose()
                raise
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.request(
                    method=request.method, url=url, headers=headers, content=body
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "connection")
                    },
                )

    return app


# Create app instance for uvicorn
app = create_app()
