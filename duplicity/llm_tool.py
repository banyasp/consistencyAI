"""Async LLM client utilities and response dataclass - FIXED VERSION for fast robust system."""

from __future__ import annotations

import time
import asyncio
import aiohttp
import ssl
import certifi
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from . import config


@dataclass
class LLMResponse:
    """Container for a single LLM response and metadata."""

    model_name: str
    provider: str
    response_text: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None

    def is_empty_or_invalid(self) -> bool:
        """Check if response is empty or clearly invalid."""
        if not self.response_text:
            return True
        if len(self.response_text.strip()) < 50:
            return True
        if not any(c.isalpha() for c in self.response_text):
            return True
        error_patterns = [
            "error", "failed", "timeout", "rate limit", "quota exceeded",
            "service unavailable", "internal server error", "bad gateway"
        ]
        response_lower = self.response_text.lower()
        if any(pattern in response_lower for pattern in error_patterns):
            return True
        return False


def get_provider_from_model(model_spec: str) -> str:
    """Extract provider prefix from a spec like 'openai/gpt-4' -> 'openai'."""

    if "/" in model_spec:
        return model_spec.split("/")[0]
    return ""


def get_model_name_from_spec(model_spec: str) -> str:
    """Extract model name from a spec like 'openai/gpt-4' -> 'gpt-4'."""

    if "/" in model_spec:
        return model_spec.split("/")[1]
    return model_spec


class LLMComparisonTool:
    """Async tool to query multiple LLMs through OpenRouter - FIXED for fast robust system."""

    def __init__(self) -> None:
        self.responses: List[LLMResponse] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time: Optional[float] = None
        self.min_request_interval: float = 0.1  # Reduced for faster processing
        self.request_count: int = 0
        self.session_start_time: Optional[float] = None
        self.session_timeout: int = 300  # 5 minutes

    async def __aenter__(self) -> "LLMComparisonTool":
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=100, limit_per_host=20)
        # Increased timeouts for reasoning models (gpt-5-pro, o1, o3) which take 60-120s to respond
        timeout = aiohttp.ClientTimeout(total=180, connect=10, sock_read=120)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Duplicity-LLM-Tool/1.0'}
        )
        self.session_start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session:
            await self.session.close()

    async def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        if self.last_request_time:
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1

    async def _check_session_health(self):
        """Check if session is still healthy and restart if needed."""
        current_time = time.time()
        
        # Restart session if it's been running too long
        if current_time - self.session_start_time > self.session_timeout:
            print(f"  Session timeout reached ({self.session_timeout}s), restarting...")
            if self.session:
                await self.session.close()
            
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context, limit=100, limit_per_host=20)
            # Increased timeouts for reasoning models (gpt-5-pro, o1, o3) which take 60-120s to respond
            timeout = aiohttp.ClientTimeout(total=180, connect=10, sock_read=120)

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'Duplicity-LLM-Tool/1.0'}
            )
            self.session_start_time = time.time()

    async def _make_request_with_retry(self, request_func, *args, **kwargs):
        """Make a request with retry logic and response validation."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                await self._rate_limit()
                await self._check_session_health()
                
                response = await request_func(*args, **kwargs)
                
                # Check if response is valid
                if hasattr(response, 'is_empty_or_invalid') and response.is_empty_or_invalid():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"  Empty/invalid response, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(f" Max retries reached, returning invalid response")
                
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Request failed: {str(e)}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    print(f" Max retries reached, returning error response")
                    # Return an error response
                    return LLMResponse(
                        model_name=kwargs.get('model_spec', 'unknown'),
                        provider='error',
                        response_text=f"Error after {max_retries} retries: {str(e)}",
                        error=str(e)
                    )
        
        return None

    async def query_via_openrouter(self, prompt: str, model_spec: str) -> LLMResponse:
        """Query a model via OpenRouter."""
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/banyasp/consistencyAI",
                "X-Title": "ConsistencyAI Benchmark"
            }
            
            data = {
                "model": model_spec,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
            
            async with self.session.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    tokens_used = result.get("usage", {}).get("total_tokens", 0)
                    
                    return LLMResponse(
                        model_name=model_spec,
                        provider="openrouter",
                        response_text=response_text,
                        tokens_used=tokens_used,
                        response_time=time.time() - start_time,
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    error_text = await response.text()
                    return LLMResponse(
                        model_name=model_spec,
                        provider="openrouter",
                        response_text="",
                        error=f"HTTP {response.status}: {error_text}",
                        response_time=time.time() - start_time,
                        timestamp=datetime.now().isoformat()
                    )
                    
        except Exception as e:
            return LLMResponse(
                model_name=model_spec,
                provider="openrouter",
                response_text="",
                error=str(e),
                response_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

    async def query_openai(self, prompt: str, model_spec: str) -> LLMResponse:
        """Query OpenAI directly."""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            model_name = get_model_name_from_spec(model_spec)

            # Models that REQUIRE the v1/responses endpoint (confirmed by API errors/failures)
            # These models ONLY work with responses endpoint, not chat/completions
            responses_only_models = [
                "gpt-5-pro-2025-10-06",  # Confirmed: returns 404 on chat/completions
                "gpt-5-pro",
                "gpt-5-nano-2025-08-07",  # Returns empty responses on chat/completions
                "gpt-5-nano",
            ]

            # Check if this model requires the responses endpoint
            use_responses_endpoint = any(required_model in model_name for required_model in responses_only_models)

            if use_responses_endpoint:
                # Use v1/responses endpoint (for reasoning/pro models)
                data = {
                    "model": model_name,
                    "input": prompt,  # responses endpoint uses "input" instead of "messages"
                    "max_output_tokens": 1000
                }
                endpoint = "https://api.openai.com/v1/responses"
            else:
                # Use standard chat completions endpoint
                data = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }

                # Some newer models require max_completion_tokens instead of max_tokens
                if "gpt-5" in model_name or "o1" in model_name or "o3" in model_name:
                    data["max_completion_tokens"] = 1000
                else:
                    data["max_tokens"] = 1000

                endpoint = "https://api.openai.com/v1/chat/completions"

            async with self.session.post(
                endpoint,
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()

                    # Parse response based on endpoint type
                    if use_responses_endpoint:
                        # v1/responses format: output_text contains the response
                        response_text = result.get("output_text", "")
                        if not response_text:
                            # Fallback: try to extract from output array
                            output = result.get("output", [])
                            if output and len(output) > 0:
                                content = output[0].get("content", [])
                                if content and len(content) > 0:
                                    response_text = content[0].get("text", "")
                    else:
                        # Standard chat completions format
                        response_text = result["choices"][0]["message"]["content"]

                    tokens_used = result.get("usage", {}).get("total_tokens", 0)

                    return LLMResponse(
                        model_name=model_spec,
                        provider="openai",
                        response_text=response_text,
                        tokens_used=tokens_used,
                        response_time=time.time() - start_time,
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    error_text = await response.text()
                    return LLMResponse(
                        model_name=model_spec,
                        provider="openai",
                        response_text="",
                        error=f"HTTP {response.status}: {error_text}",
                        response_time=time.time() - start_time,
                        timestamp=datetime.now().isoformat()
                    )

        except Exception as e:
            return LLMResponse(
                model_name=model_spec,
                provider="openai",
                response_text="",
                error=str(e),
                response_time=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )

    async def run_comparison(self, prompt: str, models: Optional[List[str]] = None, all_open_router: bool = False) -> List[LLMResponse]:
        """
        FIXED VERSION: Process a single prompt for a single model.
        The new fast robust system handles batching at a higher level.
        """
        if models is None:
            models = ["openai/gpt-4o"]

        # For the new fast system, we're processing one model at a time
        # So we don't need the old sequential processing logic
        model_spec = models[0]  # Take the first (and only) model
        
        # Route models based on provider
        # ALWAYS use direct OpenAI API for OpenAI models (never through OpenRouter)
        # This is because OpenRouter fails for ChatGPT-5 and other OpenAI models
        if "openai" in model_spec.lower():
            response = await self._make_request_with_retry(self.query_openai, prompt, model_spec)
        # For all other models (including Gemini), use OpenRouter
        else:
            response = await self._make_request_with_retry(self.query_via_openrouter, prompt, model_spec)
        
        return [response]
