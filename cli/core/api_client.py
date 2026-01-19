"""API Client for communicating with the Hybrid RAG FastAPI server."""

from __future__ import annotations

import httpx
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class APIResponse:
    """Wrapper for API responses."""
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    status_code: int = 0


class APIClient:
    """HTTP client for the Hybrid RAG API."""
    
    # Default timeout for most operations (5 minutes)
    DEFAULT_TIMEOUT = 300.0
    # Extended timeout for LLM operations that may take longer (10 minutes)
    LLM_TIMEOUT = 600.0
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._llm_client: Optional[httpx.Client] = None  # Separate client for LLM operations
    
    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client
    
    @property
    def llm_client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client with extended timeout for LLM operations."""
        if self._llm_client is None:
            self._llm_client = httpx.Client(timeout=self.LLM_TIMEOUT)
        return self._llm_client
    
    def close(self) -> None:
        """Close the HTTP clients."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._llm_client is not None:
            self._llm_client.close()
            self._llm_client = None
    
    def __enter__(self) -> "APIClient":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[dict] = None,
        files: Optional[list[tuple]] = None,
        params: Optional[dict] = None,
        use_llm_timeout: bool = False,
    ) -> APIResponse:
        """Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json: JSON body for the request
            files: Files for multipart upload
            params: Query parameters
            use_llm_timeout: Use extended timeout for LLM operations (5 min instead of 2 min)
        """
        url = f"{self.base_url}{endpoint}"
        http_client = self.llm_client if use_llm_timeout else self.client
        
        try:
            if files:
                response = http_client.request(method, url, files=files, params=params)
            else:
                response = http_client.request(method, url, json=json, params=params)
            
            if response.status_code >= 400:
                return APIResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )
            
            try:
                data = response.json()
            except Exception:
                data = {"raw": response.text}
            
            return APIResponse(success=True, data=data, status_code=response.status_code)
        
        except httpx.ConnectError:
            return APIResponse(
                success=False,
                error="Connection failed. Is the server running? Try '/start' first.",
                status_code=0,
            )
        except httpx.TimeoutException:
            timeout_val = self.LLM_TIMEOUT if use_llm_timeout else self.timeout
            return APIResponse(
                success=False,
                error=f"Request timed out after {int(timeout_val)}s. The LLM may still be processing. Try again or check server logs.",
                status_code=0,
            )
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Request failed: {type(e).__name__}: {e}",
                status_code=0,
            )
    
    # Health & Status
    def health(self) -> APIResponse:
        """Check server health."""
        return self._request("GET", "/health")
    
    def stats(self) -> APIResponse:
        """Get system statistics."""
        return self._request("GET", "/stats")
    
    # Query
    def query(
        self,
        query: str,
        session_id: str = "cli-session",
        bm25_k: int = 20,
        vec_k: int = 20,
        top_k: int = 8,
        memory_k: int = 6,
    ) -> APIResponse:
        """Send a query to the RAG system.
        
        Uses extended timeout (5 min) since LLM inference can be slow,
        especially on first use when model is being loaded.
        """
        return self._request("POST", "/query", json={
            "session_id": session_id,
            "query": query,
            "bm25_k": bm25_k,
            "vec_k": vec_k,
            "top_k": top_k,
            "memory_k": memory_k,
        }, use_llm_timeout=True)
    
    # Ingest
    def ingest(self, file_paths: list[Path]) -> APIResponse:
        """Ingest documents into the knowledge base."""
        files = []
        for path in file_paths:
            if path.exists():
                files.append(("files", (path.name, open(path, "rb"), "application/octet-stream")))
        
        if not files:
            return APIResponse(success=False, error="No valid files to ingest")
        
        try:
            result = self._request("POST", "/ingest", files=files, use_llm_timeout=True)
        finally:
            # Close file handles
            for _, (_, fh, _) in files:
                fh.close()
        
        return result
    
    # Debug
    def debug_retrieval(
        self,
        query: str,
        bm25_k: int = 20,
        vec_k: int = 20,
        top_k: int = 8,
    ) -> APIResponse:
        """Debug retrieval (BM25, vector, fused results)."""
        return self._request("POST", "/debug/retrieval", json={
            "query": query,
            "bm25_k": bm25_k,
            "vec_k": vec_k,
            "top_k": top_k,
        })
    
    def debug_citations(
        self,
        query: str,
        session_id: str = "debug",
        bm25_k: int = 20,
        vec_k: int = 20,
        top_k: int = 8,
        bm25_mode: str = "heuristic",
    ) -> APIResponse:
        """Debug citations. Uses extended timeout for LLM operations."""
        return self._request("POST", "/debug/citations", json={
            "query": query,
            "session_id": session_id,
            "bm25_k": bm25_k,
            "vec_k": vec_k,
            "top_k": top_k,
            "bm25_mode": bm25_mode,
        }, use_llm_timeout=True)
    
    # Chunks
    def get_chunk(self, chunk_id: int) -> APIResponse:
        """Get a specific chunk by ID."""
        return self._request("GET", f"/chunks/{chunk_id}")
