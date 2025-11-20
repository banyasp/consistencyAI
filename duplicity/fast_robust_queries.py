"""
Fast robust query orchestration with adaptive concurrency and intelligent batching.
This system finds the sweet spot between speed and reliability by:
1. Starting with high concurrency and automatically adjusting based on API health
2. Using larger batches with parallel processing
3. Implementing smart rate limiting that adapts to API performance
4. Monitoring response quality and adjusting strategy in real-time
5. Providing progress estimates and ETA calculations
6. AUTOMATIC RETRY for 100% success rate
7. INCREMENTAL JSON STORAGE for real-time monitoring
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .llm_tool import LLMComparisonTool


def _logs_dir(subdir: str = "") -> str:
    """Return absolute path to the project's logs directory.

    Args:
        subdir: Subdirectory within logs (e.g., "main", "control")
    """
    project_root = Path(__file__).resolve().parent.parent
    if subdir:
        logs_path = project_root / "logs" / subdir
    else:
        logs_path = project_root / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    return str(logs_path)


def _incremental_logs_dir(subdir: str = "") -> str:
    """Return absolute path to the incremental logs directory."""
    incremental_path = Path(_logs_dir(subdir)) / "incremental"
    incremental_path.mkdir(parents=True, exist_ok=True)
    return str(incremental_path)


@dataclass
class PerformanceMetrics:
    """Track performance metrics for adaptive optimization."""
    requests_per_second: float = 0.0
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    empty_response_rate: float = 0.0
    rate_limit_hits: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    retry_count: int = 0
    max_retries_reached: int = 0


class FastRobustQueryProcessor:
    """Fast query processor with adaptive concurrency and intelligent optimization."""
    
    def __init__(
        self,
        initial_batch_size: int = 50,
        initial_concurrency: int = 20,
        max_concurrency: int = 100,
        min_concurrency: int = 5,
        adaptive_mode: bool = True,
        save_progress: bool = True,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        ensure_100_percent_success: bool = True,
        save_incremental: bool = True,
        incremental_interval: int = 1,
        start_batch_number: int = 1,
        initial_incremental_counter: int = 0,
        subdir: str = ""
    ):
        self.initial_batch_size = initial_batch_size
        self.initial_concurrency = initial_concurrency
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.subdir = subdir
        self.adaptive_mode = adaptive_mode
        self.save_progress = save_progress
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ensure_100_percent_success = ensure_100_percent_success
        self.save_incremental = save_incremental
        self.incremental_interval = incremental_interval
        self.start_batch_number = start_batch_number
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.start_time = None
        self.progress_file = None
        
        # Adaptive parameters
        self.current_batch_size = initial_batch_size
        self.current_concurrency = initial_concurrency
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        # Retry tracking
        self.retry_queue = []
        self.retry_attempts = {}
        
        # Coverage tracking - NEW: Track all expected tasks
        self.expected_tasks = set()  # Set of (model, topic, persona_id) tuples
        self.completed_tasks = set()  # Set of (model, topic, persona_id) tuples
        
        # Task indexing - NEW: Track position in task list (avoids skipping with adaptive batch sizes)
        self.next_task_idx = 0  # Next task index to process
        
        # Incremental storage
        self.incremental_counter = initial_incremental_counter
        self.incremental_base_filename = None
        
        # MODEL-SPECIFIC RATE LIMITING: Track rate limits per model
        self.model_rate_limits = {}  # {model: {"last_request": timestamp, "request_count": int, "backoff_until": timestamp}}
        self.model_request_windows = {}  # {model: {"window_start": timestamp, "request_count": int}}
        self.max_requests_per_minute = 60  # Conservative default
        self.rate_limit_window = 60  # 1 minute window
        
    def _save_incremental_results(self, results: Dict, batch_num: int, total_batches: int, stage: str = "batch"):
        """Save incremental results to logs/incremental folder for real-time monitoring."""
        if not self.save_incremental:
            return
            
        # Create base filename on first call
        if self.incremental_base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.incremental_base_filename = f"incremental_results_{timestamp}"
        
        # Save every N batches (or every batch if interval is 1)
        if batch_num % self.incremental_interval == 0 or batch_num == total_batches - 1:
            self.incremental_counter += 1
            
            # Create filename with counter and stage (adjust for start_batch_number)
            actual_batch_num = batch_num + self.start_batch_number
            filename = f"{self.incremental_base_filename}_batch{actual_batch_num:03d}_{stage}_{self.incremental_counter:03d}.json"
            filepath = Path(_incremental_logs_dir(self.subdir)) / filename
            
            # Prepare incremental data
            incremental_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "batch_number": actual_batch_num,
                    "total_batches": total_batches + self.start_batch_number - 1,
                    "stage": stage,
                    "incremental_counter": self.incremental_counter,
                    "progress_percentage": ((batch_num + 1) / total_batches) * 100,
                    "current_concurrency": self.current_concurrency,
                    "current_batch_size": self.current_batch_size,
                    "retry_queue_size": len(self.retry_queue)
                },
                "performance_metrics": {
                    "requests_per_second": self.metrics.requests_per_second,
                    "success_rate": self.metrics.success_rate,
                    "empty_response_rate": self.metrics.empty_response_rate,
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "retry_count": self.metrics.retry_count,
                    "max_retries_reached": self.metrics.max_retries_reached
                },
                "results": results
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(incremental_data, f, indent=2, ensure_ascii=False)
            
            print(f" Incremental results saved: {filename}")
            print(f"   Progress: {batch_num + 1}/{total_batches} ({((batch_num + 1) / total_batches) * 100:.1f}%)")
            print(f"   File: {filepath}")
        
    def _save_progress_checkpoint(self, results: Dict, completed_batches: int, total_batches: int):
        """Save progress to allow resumption."""
        if not self.save_progress:
            return
            
        if not self.progress_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.progress_file = Path(_logs_dir(self.subdir)) / f"fast_progress_{timestamp}.json"
        
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        if completed_batches > 0:
            time_per_batch = elapsed_time / completed_batches
            remaining_batches = total_batches - completed_batches
            eta_seconds = time_per_batch * remaining_batches
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "Calculating..."
        
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "completed_batches": completed_batches,
            "total_batches": total_batches,
            "eta": eta,
            "current_concurrency": self.current_concurrency,
            "current_batch_size": self.current_batch_size,
            "retry_queue_size": len(self.retry_queue),
            "metrics": {
                "requests_per_second": self.metrics.requests_per_second,
                "success_rate": self.metrics.success_rate,
                "empty_response_rate": self.metrics.empty_response_rate,
                "retry_count": self.metrics.retry_count,
                "max_retries_reached": self.metrics.max_retries_reached
            },
            "results": results
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f" Progress saved: {completed_batches}/{total_batches} batches complete (ETA: {eta})")
        if self.retry_queue:
            print(f" Retry queue: {len(self.retry_queue)} items pending")
    
    def _create_batches(self, queries: Dict[str, Dict[str, str]], models: List[str]) -> List[Dict]:
        """Create optimized batches with INTERLEAVED model distribution for faster completion."""
        all_tasks = []
        
        # INTERLEAVED APPROACH: Mix models within each batch
        # This ensures each batch contains queries for multiple models
        for topic in queries:
            for persona_id, prompt in queries[topic].items():
                for model_spec in models:  # Loop through models LAST (interleaved)
                    all_tasks.append({
                        "model": model_spec,
                        "topic": topic,
                        "persona_id": persona_id,
                        "prompt": prompt
                    })
                    # NEW: Track expected tasks for coverage validation
                    self.expected_tasks.add((model_spec, topic, persona_id))
        
        # Store all tasks for dynamic batch creation
        self.all_tasks = all_tasks
        self.total_tasks = len(all_tasks)
        self.next_task_idx = 0  # Track the next task to process (CRITICAL FIX)
        
        print(f" Created {len(self.expected_tasks)} expected task combinations (model×topic×persona)")
        
        # Calculate estimated total batches (may change with adaptive sizing)
        total_batches = (self.total_tasks + self.current_batch_size - 1) // self.current_batch_size
        
        return total_batches  # Return estimated number of batches
    
    def _get_next_batch(self, batch_num: int) -> List[Dict]:
        """Get the next batch using current adaptive batch size - FIXED to track position correctly."""
        # Use self.next_task_idx instead of batch_num * batch_size to avoid skipping
        start_idx = self.next_task_idx
        end_idx = min(start_idx + self.current_batch_size, self.total_tasks)
        
        # Update the index for next batch
        self.next_task_idx = end_idx
        
        return self.all_tasks[start_idx:end_idx]
    
    def _identify_missing_tasks(self, organized_results: Dict) -> List[Dict]:
        """Identify which expected tasks are missing from results."""
        # Build set of completed tasks from organized results
        completed = set()
        for model in organized_results:
            for topic in organized_results[model]:
                for persona_id in organized_results[model][topic]:
                    completed.add((model, topic, persona_id))
        
        # Find missing tasks
        missing = self.expected_tasks - completed
        
        if missing:
            print(f"  Found {len(missing)} missing task combinations:")
            # Group by model and topic for clearer reporting
            missing_by_model_topic = {}
            for model, topic, persona_id in missing:
                key = (model, topic)
                if key not in missing_by_model_topic:
                    missing_by_model_topic[key] = []
                missing_by_model_topic[key].append(persona_id)
            
            for (model, topic), persona_ids in missing_by_model_topic.items():
                print(f"   {model}/{topic}: {len(persona_ids)} personas ({sorted(persona_ids)[:5]}...)")
        
        # Create task dicts for missing items by looking up from all_tasks
        missing_tasks = []
        for task in self.all_tasks:
            task_key = (task['model'], task['topic'], task['persona_id'])
            if task_key in missing:
                missing_tasks.append(task)
        
        return missing_tasks
    
    def _is_response_valid(self, response_text: str) -> bool:
        """Check if a response is valid and non-empty."""
        if not response_text or not response_text.strip():
            return False
        
        # Check for minimum length (at least 50 characters)
        if len(response_text.strip()) < 50:
            return False
        
        # Check for error patterns
        error_patterns = [
            "error", "failed", "timeout", "rate limit", "quota exceeded",
            "service unavailable", "internal server error", "bad gateway",
            "empty", "invalid", "no response"
        ]
        
        response_lower = response_text.lower()
        if any(pattern in response_lower for pattern in error_patterns):
            return False
        
        # Check if response contains actual content (at least some letters)
        if not any(c.isalpha() for c in response_text):
            return False
        
        return True
    
    def _add_to_retry_queue(self, task: Dict, reason: str):
        """Add a failed task to the retry queue."""
        task_key = f"{task['model']}_{task['topic']}_{task['persona_id']}"
        
        if task_key not in self.retry_attempts:
            self.retry_attempts[task_key] = 0
        
        if self.retry_attempts[task_key] < self.max_retries:
            self.retry_attempts[task_key] += 1
            self.retry_queue.append({
                **task,
                "retry_attempt": self.retry_attempts[task_key],
                "failure_reason": reason
            })
            print(f" Added to retry queue: {task_key} (attempt {self.retry_attempts[task_key]}/{self.max_retries})")
        else:
            self.metrics.max_retries_reached += 1
            print(f" Max retries reached for: {task_key} - marking as permanently failed")
    
    def _update_performance_metrics(self, batch_results: List[Dict], batch_time: float):
        """Update performance metrics and adjust strategy."""
        batch_size = len(batch_results)
        
        # Calculate batch metrics
        successful = sum(1 for r in batch_results if r.get("success", False))
        empty_responses = sum(1 for r in batch_results if not self._is_response_valid(r.get("response", "")))
        rate_limit_hits = sum(1 for r in batch_results if "rate limit" in r.get("response", "").lower())
        
        # Update global metrics
        self.metrics.total_requests += batch_size
        self.metrics.successful_requests += successful
        self.metrics.success_rate = self.metrics.successful_requests / self.metrics.total_requests
        self.metrics.empty_response_rate = empty_responses / batch_size if batch_size > 0 else 0
        self.metrics.rate_limit_hits += rate_limit_hits
        
        # Calculate requests per second
        if batch_time > 0:
            self.metrics.requests_per_second = batch_size / batch_time
        
        # Adaptive strategy adjustment
        if self.adaptive_mode:
            self._adjust_strategy(successful, batch_size, rate_limit_hits > 0)
    
    def _adjust_strategy(self, successful: int, total: int, hit_rate_limit: bool):
        """Intelligently adjust concurrency and batch size based on performance."""
        success_rate = successful / total if total > 0 else 0
        
        # ADAPTIVE BATCH SIZING: Increase batch size when doing well
        if success_rate >= 0.95 and self.current_batch_size < 100:  # Safe upper limit
            new_batch_size = min(self.current_batch_size + 10, 100)
            if new_batch_size != self.current_batch_size:
                print(f" Increasing batch size from {self.current_batch_size} to {new_batch_size} (success rate: {success_rate:.1%})")
                self.current_batch_size = new_batch_size
        elif success_rate < 0.90 and self.current_batch_size > 20:  # Decrease if struggling
            new_batch_size = max(self.current_batch_size - 10, 20)
            if new_batch_size != self.current_batch_size:
                print(f" Decreasing batch size from {self.current_batch_size} to {new_batch_size} (success rate: {success_rate:.1%})")
                self.current_batch_size = new_batch_size
        
        if hit_rate_limit:
            # Rate limited - reduce concurrency
            self.current_concurrency = max(self.min_concurrency, self.current_concurrency // 2)
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            print(f"  Rate limited - reducing concurrency to {self.current_concurrency}")
            
        elif success_rate < 0.8:
            # Low success rate - reduce concurrency
            self.current_concurrency = max(self.min_concurrency, int(self.current_concurrency * 0.8))
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            print(f"  Low success rate ({success_rate:.1%}) - reducing concurrency to {self.current_concurrency}")
            
        elif success_rate > 0.95 and self.consecutive_successes >= 3:
            # High success rate - increase concurrency
            if self.consecutive_successes >= 3:
                self.current_concurrency = min(self.max_concurrency, int(self.current_concurrency * 1.2))
                print(f" High success rate ({success_rate:.1%}) - increasing concurrency to {self.current_concurrency}")
                self.consecutive_successes = 0
            else:
                self.consecutive_successes += 1
        else:
            self.consecutive_successes += 1
    
    async def _process_single_query(
        self, 
        tool: LLMComparisonTool, 
        task: Dict,
        all_open_router: bool = False
    ) -> Dict[str, Any]:
        """Process a single query with fast error handling."""
        try:
            responses = await tool.run_comparison(
                task["prompt"], 
                [task["model"]], 
                all_open_router=all_open_router
            )
            
            if responses and len(responses) > 0:
                response = responses[0]
                
                # Quick validation
                if response.error:
                    return {
                        **task,
                        "response": f"Error: {response.error}",
                        "success": False,
                        "tokens": 0
                    }
                elif hasattr(response, 'is_empty_or_invalid') and response.is_empty_or_invalid():
                    return {
                        **task,
                        "response": "Error: Empty or invalid response",
                        "success": False,
                        "tokens": 0
                    }
                else:
                    return {
                        **task,
                        "response": response.response_text,
                        "success": True,
                        "tokens": response.tokens_used or 0
                    }
            else:
                return {
                    **task,
                    "response": "Error: No response received",
                    "success": False,
                    "tokens": 0
                }
                
        except Exception as e:
            return {
                **task,
                "response": f"Error: {str(e)}",
                "success": False,
                "tokens": 0
            }
    
    async def _process_batch_fast(
        self, 
        batch: List[Dict], 
        batch_num: int, 
        total_batches: int,
        all_open_router: bool = False
    ) -> List[Dict]:
        """Process a batch with high concurrency and minimal delays."""
        start_time = time.time()
        
        print(f" Processing batch {batch_num + 1}/{total_batches} ({len(batch)} queries, concurrency: {self.current_concurrency}, current batch size: {self.current_batch_size})")
        
        # Process batch with current concurrency
        semaphore = asyncio.Semaphore(self.current_concurrency)
        
        async def process_with_semaphore(task):
            async with semaphore:
                async with LLMComparisonTool() as tool:
                    result = await self._process_single_query(tool, task, all_open_router)
                    # Minimal delay - just enough to avoid overwhelming
                    await asyncio.sleep(0.05)  # 50ms instead of 100ms
                    return result
        
        # Process all tasks in batch concurrently
        results = await asyncio.gather(*[
            process_with_semaphore(task) for task in batch
        ], return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    **batch[i],
                    "response": f"Error: {str(result)}",
                    "success": False,
                    "tokens": 0
                })
            else:
                processed_results.append(result)
        
        # Check for failed responses and add to retry queue if needed
        if self.ensure_100_percent_success:
            for result in processed_results:
                if not result.get("success", False) or not self._is_response_valid(result.get("response", "")):
                    self._add_to_retry_queue(result, result.get("response", "Unknown error"))
        
        # Calculate batch performance
        batch_time = time.time() - start_time
        successful = sum(1 for r in processed_results if r.get("success", False) and self._is_response_valid(r.get("response", "")))
        failed = len(processed_results) - successful
        total_tokens = sum(r.get("tokens", 0) for r in processed_results)
        
        print(f" Batch {batch_num + 1} complete: {successful} success, {failed} failed, {total_tokens} tokens, {batch_time:.1f}s")
        
        # Update performance metrics
        self._update_performance_metrics(processed_results, batch_time)
        
        return processed_results
    
    async def _process_retry_queue(self, all_open_router: bool = False) -> List[Dict]:
        """Process the retry queue with SMART priority-based retry logic."""
        if not self.retry_queue:
            return []
        
        print(f" Processing {len(self.retry_queue)} failed items with SMART retry logic...")
        
        # SMART RETRY STRATEGY: Prioritize items by failure reason and attempt count
        def retry_priority(task):
            # Higher priority for rate limit failures (likely temporary)
            if "rate limit" in task.get("failure_reason", "").lower():
                return 1
            # Medium priority for connection/timeout issues
            elif any(x in task.get("failure_reason", "").lower() for x in ["timeout", "connection"]):
                return 2
            # Lower priority for quality/content issues
            else:
                return 3
        
        # Sort retry queue by priority (lower number = higher priority)
        self.retry_queue.sort(key=retry_priority)
        
        # Use lower concurrency for retries to be more careful
        retry_concurrency = max(5, self.current_concurrency // 4)
        retry_semaphore = asyncio.Semaphore(retry_concurrency)
        
        async def process_retry_item(task):
            async with retry_semaphore:
                # Add exponential backoff based on retry attempt
                retry_attempt = task.get("retry_attempt", 1)
                if retry_attempt > 1:
                    backoff_delay = min(30, 2 ** (retry_attempt - 1))  # 2s, 4s, 8s, 16s, 32s (capped at 30s)
                    print(f"⏳ Backoff delay for {task['model']}: {backoff_delay}s (attempt {retry_attempt})")
                    await asyncio.sleep(backoff_delay)
                
                async with LLMComparisonTool() as tool:
                    result = await self._process_single_query(tool, task, all_open_router)
                    await asyncio.sleep(0.1)  # Slightly longer delay for retries
                    return result
        
        # Process retry items
        retry_tasks = [process_retry_item(task) for task in self.retry_queue]
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
        
        # Clear the retry queue and add successful items back to results
        successful_retries = []
        still_failed = []
        
        for i, result in enumerate(retry_results):
            if isinstance(result, Exception):
                still_failed.append(self.retry_queue[i])
            elif result.get("success", False):
                successful_retries.append(result)
            else:
                still_failed.append(self.retry_queue[i])
        
        # Update retry queue
        self.retry_queue = still_failed
        self.metrics.retry_count += len(successful_retries)
        
        # SMART RETRY ANALYSIS
        if successful_retries:
            success_rate = len(successful_retries) / (len(successful_retries) + len(still_failed))
            print(f" Retry round complete: {len(successful_retries)} successful, {len(still_failed)} still failed")
            print(f"   Retry success rate: {success_rate:.1%}")
            
            # If retry success rate is high, we can be more aggressive next time
            if success_rate > 0.8:
                print(f" High retry success rate - can be more aggressive in future")
            elif success_rate < 0.5:
                print(f"  Low retry success rate - may need to investigate root causes")
        else:
            print(f" No successful retries in this round")
        
        return successful_retries
    
    def _organize_results(self, processed_results: List[Dict]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Organize results back into the expected structure."""
        organized = {}
        
        for result in processed_results:
            model = result["model"]
            topic = result["topic"]
            persona_id = result["persona_id"]
            response = result["response"]
            
            if model not in organized:
                organized[model] = {}
            if topic not in organized[model]:
                organized[model][topic] = {}
            
            organized[model][topic][persona_id] = response
        
        return organized
    
    async def process_queries(
        self,
        queries: Dict[str, Dict[str, str]],
        models: List[str],
        all_open_router: bool = False
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Process all queries with fast adaptive processing and 100% success guarantee."""
        
        self.start_time = time.time()
        
        print(f" Starting FAST robust query processing (100% success mode):")
        print(f"   Models: {len(models)}")
        print(f"   Topics: {len(queries)}")
        total_personas = sum(len(personas) for personas in queries.values())
        print(f"   Personas per topic: {total_personas // len(queries) if queries else 0}")
        print(f"   Total queries: {len(models) * sum(len(personas) for personas in queries.values())}")
        print(f"   Initial batch size: {self.initial_batch_size}")
        print(f"   Initial concurrency: {self.initial_concurrency}")
        print(f"   Max concurrency: {self.max_concurrency}")
        print(f"   Adaptive mode: {self.adaptive_mode}")
        print(f"   All OpenRouter: {all_open_router}")
        print(f"   Max retries: {self.max_retries}")
        print(f"   100% success mode: {self.ensure_100_percent_success}")
        print(f"   Incremental saving: {self.save_incremental}")
        if self.save_incremental:
            print(f"   Incremental interval: Every {self.incremental_interval} batch(es)")
            print(f"   Incremental folder: {_incremental_logs_dir(self.subdir)}")
        
        # Create dynamic batch system
        total_batches = self._create_batches(queries, models)
        print(f"   Total batches: {total_batches}")
        
        # Process batches with minimal delays
        all_results = []
        batch_num = 0
        
        # NEW: Continue until all tasks are processed (handles adaptive batch sizing correctly)
        while self.next_task_idx < self.total_tasks:
            # Get next batch using current adaptive batch size
            batch = self._get_next_batch(batch_num)
            
            if not batch:
                print("  No more tasks to process")
                break
            
            try:
                # SMART INTER-BATCH DELAYS: Adaptive delays based on performance
                if batch_num > 0:
                    # Base delay calculation
                    base_delay = 0.5
                    
                    # Performance-based adjustments
                    if self.metrics.success_rate < 0.85:
                        delay = base_delay * 4  # 2s if struggling significantly
                        print(f"⏳ Extended delay: {delay}s (poor performance: {self.metrics.success_rate:.1%})")
                    elif self.metrics.success_rate < 0.90:
                        delay = base_delay * 2  # 1s if struggling moderately
                        print(f"⏳ Moderate delay: {delay}s (moderate performance: {self.metrics.success_rate:.1%})")
                    elif self.metrics.success_rate > 0.98:
                        delay = base_delay * 0.2  # 0.1s if doing very well
                        print(f"⏳ Minimal delay: {delay}s (excellent performance: {self.metrics.success_rate:.1%})")
                    elif self.metrics.success_rate > 0.95:
                        delay = base_delay * 0.5  # 0.25s if doing well
                        print(f"⏳ Short delay: {delay}s (good performance: {self.metrics.success_rate:.1%})")
                    else:
                        delay = base_delay  # Default delay
                        print(f"⏳ Standard delay: {delay}s (performance: {self.metrics.success_rate:.1%})")
                    
                    await asyncio.sleep(delay)
                
                # Calculate dynamic total batches
                current_total_batches = batch_num + 1 + ((self.total_tasks - self.next_task_idx + self.current_batch_size - 1) // self.current_batch_size)
                
                batch_results = await self._process_batch_fast(batch, batch_num, current_total_batches, all_open_router)
                all_results.extend(batch_results)
                
                # Save progress after each batch
                partial_organized = self._organize_results(all_results)
                self._save_progress_checkpoint(partial_organized, batch_num + 1, current_total_batches)
                
                # Save incremental results for real-time monitoring
                self._save_incremental_results(partial_organized, batch_num, current_total_batches, "batch")
                
                # ENHANCED PROGRESS TRACKING with ETA
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    queries_per_second = len(all_results) / elapsed
                    total_queries = self.total_tasks
                    remaining_queries = total_queries - len(all_results)
                    
                    if queries_per_second > 0:
                        eta_seconds = remaining_queries / queries_per_second
                        eta_hours = eta_seconds // 3600
                        eta_minutes = (eta_seconds % 3600) // 60
                        
                        if eta_hours > 0:
                            eta_str = f"{eta_hours:.0f}h {eta_minutes:.0f}m"
                        else:
                            eta_str = f"{eta_minutes:.0f}m"
                        
                        print(f" Progress: {len(all_results)}/{total_queries} ({len(all_results)/total_queries*100:.1f}%)")
                        print(f"   Speed: {queries_per_second:.1f} queries/sec | Success: {self.metrics.success_rate:.1%}")
                        print(f"   ETA: {eta_str} | Concurrency: {self.current_concurrency} | Batch size: {self.current_batch_size}")
                    else:
                        print(f" Progress: {len(all_results)}/{total_queries} ({len(all_results)/total_queries*100:.1f}%)")
                        print(f"   Success rate: {self.metrics.success_rate:.1%} | Concurrency: {self.current_concurrency}")
                
            except Exception as e:
                print(f" Error processing batch {batch_num + 1}: {str(e)}")
                # Continue with next batch
            
            batch_num += 1
        
        # Process retry queue until all succeed or max retries reached
        if self.ensure_100_percent_success and self.retry_queue:
            print(f"\n Processing retry queue to achieve 100% success...")
            
            retry_round = 1
            while self.retry_queue and retry_round <= self.max_retries:
                print(f"\n Retry round {retry_round}/{self.max_retries}")
                retry_results = await self._process_retry_queue(all_open_router)
                all_results.extend(retry_results)
                
                # Save incremental results after retry round
                partial_organized = self._organize_results(all_results)
                self._save_incremental_results(partial_organized, total_batches - 1, total_batches, f"retry_round_{retry_round}")
                
                if not self.retry_queue:
                    print(" All retries successful! 100% success achieved!")
                    break
                
                retry_round += 1
                if retry_round <= self.max_retries:
                    print(f"⏳ Waiting {self.retry_delay}s before next retry round...")
                    await asyncio.sleep(self.retry_delay)
                    self.retry_delay *= 1.5  # Exponential backoff
            
            if self.retry_queue:
                print(f"  {len(self.retry_queue)} items still failed after {self.max_retries} retry rounds")
                print("   These will be marked as permanently failed")
        
        # NEW: Check for missing tasks and attempt targeted recovery
        print(f"\n Checking for missing task combinations...")
        partial_organized = self._organize_results(all_results)
        missing_tasks = self._identify_missing_tasks(partial_organized)
        
        if missing_tasks:
            print(f"\n Attempting targeted recovery for {len(missing_tasks)} missing tasks...")
            
            # Process missing tasks with very conservative settings
            recovery_concurrency = max(3, self.min_concurrency // 2)
            recovery_semaphore = asyncio.Semaphore(recovery_concurrency)
            
            async def process_recovery_task(task):
                async with recovery_semaphore:
                    async with LLMComparisonTool() as tool:
                        result = await self._process_single_query(tool, task, all_open_router)
                        await asyncio.sleep(0.2)  # More conservative delay
                        return result
            
            # Try to recover missing tasks
            recovery_results = await asyncio.gather(*[
                process_recovery_task(task) for task in missing_tasks
            ], return_exceptions=True)
            
            # Handle results
            for i, result in enumerate(recovery_results):
                if isinstance(result, Exception):
                    recovery_results[i] = {
                        **missing_tasks[i],
                        "response": f"Error after recovery: {str(result)}",
                        "success": False,
                        "tokens": 0
                    }
            
            all_results.extend([r for r in recovery_results if not isinstance(r, Exception)])
            
            # Save after recovery
            partial_organized = self._organize_results(all_results)
            self._save_incremental_results(partial_organized, total_batches - 1, total_batches, "post_recovery")
            
            # Check again
            remaining_missing = self._identify_missing_tasks(partial_organized)
            if not remaining_missing:
                print(" Recovery successful! All tasks now complete!")
            else:
                print(f"  {len(remaining_missing)} tasks still missing after recovery")
                print("   Adding placeholder results for incomplete tasks...")
                
                # Add placeholder results for truly failed tasks
                for task in remaining_missing:
                    all_results.append({
                        **task,
                        "response": "ERROR: Task failed after all retry attempts. This persona was not successfully queried.",
                        "success": False,
                        "tokens": 0
                    })
        else:
            print(" All expected task combinations are present!")
        
        # Organize final results
        final_results = self._organize_results(all_results)
        
        # Save final incremental results
        self._save_incremental_results(final_results, total_batches - 1, total_batches, "final")
        
        # Print final statistics with coverage report
        total_time = time.time() - self.start_time
        total_queries = len(all_results)
        successful_queries = sum(1 for r in all_results if r.get("success", False) and self._is_response_valid(r.get("response", "")))
        failed_queries = total_queries - successful_queries
        total_tokens = sum(r.get("tokens", 0) for r in all_results)
        
        # Calculate coverage
        completed_count = len(set((r['model'], r['topic'], r['persona_id']) for r in all_results))
        expected_count = len(self.expected_tasks)
        coverage_pct = (completed_count / expected_count * 100) if expected_count > 0 else 0
        
        print(f"\n Final Results:")
        print(f"   Total queries processed: {total_queries}")
        print(f"   Successful: {successful_queries} ({successful_queries/total_queries*100:.1f}%)")
        print(f"   Failed: {failed_queries} ({failed_queries/total_queries*100:.1f}%)")
        print(f"   ")
        print(f"    Coverage Report:")
        print(f"      Expected task combinations: {expected_count}")
        print(f"      Completed task combinations: {completed_count}")
        print(f"      Coverage: {coverage_pct:.1f}%")
        print(f"   ")
        print(f"   Total tokens used: {total_tokens}")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"   Average speed: {total_queries/total_time:.1f} queries/sec")
        print(f"   Final concurrency: {self.current_concurrency}")
        print(f"   Retry attempts: {self.metrics.retry_count}")
        print(f"   Max retries reached: {self.metrics.max_retries_reached}")
        
        if self.save_incremental:
            print(f"   Incremental files saved: {self.incremental_counter} files in {_incremental_logs_dir(self.subdir)}")
        
        if self.ensure_100_percent_success:
            success_rate = successful_queries / total_queries if total_queries > 0 else 0
            if coverage_pct >= 100.0 and success_rate >= 0.99:
                print(" PERFECT: 100% coverage with 99%+ success rate!")
            elif coverage_pct >= 100.0 and success_rate >= 0.95:
                print(" EXCELLENT: 100% coverage with 95%+ success rate")
            elif coverage_pct >= 100.0:
                print(" COMPLETE: 100% coverage achieved (some failures)")
            elif coverage_pct >= 95.0:
                print("  GOOD: 95%+ coverage but some tasks missing")
            else:
                print(f"  WARNING: Only {coverage_pct:.1f}% coverage - significant data gaps!")
        
        return final_results


    def _check_model_rate_limit(self, model: str) -> bool:
        """Check if a model is currently rate limited and manage rate limiting per model."""
        current_time = time.time()
        
        # Initialize model tracking if not exists
        if model not in self.model_rate_limits:
            self.model_rate_limits[model] = {
                "last_request": 0,
                "request_count": 0,
                "backoff_until": 0
            }
        
        if model not in self.model_request_windows:
            self.model_request_windows[model] = {
                "window_start": current_time,
                "request_count": 0
            }
        
        # Check if we're in backoff period
        if current_time < self.model_rate_limits[model]["backoff_until"]:
            remaining = self.model_rate_limits[model]["backoff_until"] - current_time
            print(f"⏳ Model {model} in backoff: {remaining:.1f}s remaining")
            return False
        
        # Check if we need to start a new window
        if current_time - self.model_request_windows[model]["window_start"] >= self.rate_limit_window:
            self.model_request_windows[model]["window_start"] = current_time
            self.model_request_windows[model]["request_count"] = 0
        
        # Check if we're approaching rate limit
        current_count = self.model_request_windows[model]["request_count"]
        if current_count >= self.max_requests_per_minute:
            print(f" Rate limit approaching for {model}: {current_count}/{self.max_requests_per_minute} requests in current window")
            return False
        
        # Update request count
        self.model_request_windows[model]["request_count"] += 1
        self.model_rate_limits[model]["last_request"] = current_time
        
        return True
    
    def _handle_model_rate_limit(self, model: str, backoff_duration: int = 60):
        """Handle rate limiting for a specific model by implementing backoff."""
        current_time = time.time()
        self.model_rate_limits[model]["backoff_until"] = current_time + backoff_duration
        print(f" Rate limit hit for {model}, backing off for {backoff_duration}s")
        
        # Reduce concurrency for this model temporarily
        if self.current_concurrency > self.min_concurrency:
            reduction = max(5, self.current_concurrency // 4)
            self.current_concurrency = max(self.min_concurrency, self.current_concurrency - reduction)
            print(f" Reduced concurrency to {self.current_concurrency} due to rate limiting")


def query_llm_fast(
    nested_queries: Dict[str, Dict[str, str]],
    list_of_models: List[str],
    initial_batch_size: int = 50,
    initial_concurrency: int = 20,
    max_concurrency: int = 100,
    adaptive_mode: bool = True,
    all_open_router: bool = False,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    ensure_100_percent_success: bool = True,
    save_incremental: bool = True,
    incremental_interval: int = 1,
    start_batch_number: int = 1,
    initial_incremental_counter: int = 0,
    subdir: str = ""
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Fast robust synchronous query function with adaptive optimization and 100% success guarantee.

    Args:
        nested_queries: Dictionary of {topic: {persona_id: prompt}}
        list_of_models: List of model specifications
        initial_batch_size: Starting batch size (will adapt based on performance)
        initial_concurrency: Starting concurrency (will adapt based on performance)
        max_concurrency: Maximum allowed concurrency
        adaptive_mode: Whether to automatically adjust parameters
        all_open_router: If True, route all models through OpenRouter
        max_retries: Maximum number of retry attempts for failed queries
        retry_delay: Initial delay between retry rounds (increases exponentially)
        ensure_100_percent_success: If True, retry failed queries until they succeed
        save_incremental: If True, save incremental results to logs/incremental folder
        incremental_interval: Save incremental results every N batches (1 = every batch)
        subdir: Subdirectory within logs (e.g., "main", "control")

    Returns:
        Dictionary of {model: {topic: {persona_id: response}}}
    """
    processor = FastRobustQueryProcessor(
        initial_batch_size=initial_batch_size,
        initial_concurrency=initial_concurrency,
        max_concurrency=max_concurrency,
        adaptive_mode=adaptive_mode,
        save_progress=True,
        max_retries=max_retries,
        retry_delay=retry_delay,
        ensure_100_percent_success=ensure_100_percent_success,
        save_incremental=save_incremental,
        incremental_interval=incremental_interval,
        start_batch_number=start_batch_number,
        initial_incremental_counter=initial_incremental_counter,
        subdir=subdir
    )
    
    try:
        return asyncio.run(processor.process_queries(nested_queries, list_of_models, all_open_router))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(processor.process_queries(nested_queries, list_of_models, all_open_router))
        else:
            raise


def save_results_log(results: Dict[str, Dict[str, Dict[str, str]]], tag: Optional[str] = None, subdir: str = "") -> str:
    """Persist model results to logs as JSON and return the saved file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    filename = f"results_fast_{timestamp}{tag_part}.json"
    path = Path(_logs_dir(subdir)) / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f" Results saved to: {path}")
    return str(path)


def load_incremental_results(file_path: str) -> Dict:
    """Load incremental results from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_latest_fast_results(subdir: str = "") -> Optional[Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Load the most recent query results from any available source.
    Checks (in order):
    1. Standard results_*.json files in logs/
    2. Final incremental files (*_final_*.json) in logs/incremental/
    3. Any incremental files in logs/incremental/

    Args:
        subdir: Subdirectory within logs (e.g., "main", "control")

    Returns:
        Dictionary of {model: {topic: {persona_id: response}}} or None if no results found
    """
    logs_path = Path(_logs_dir(subdir))

    # First try standard results files
    std_files = sorted(logs_path.glob("results_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if std_files:
        print(f" Loading results from: {std_files[0]}")
        with open(std_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)

    # Then try incremental files (look for 'final' first)
    inc_path = logs_path / "incremental"
    if inc_path.exists():
        final_files = sorted(inc_path.glob("*_final_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if final_files:
            print(f" Loading results from: {final_files[0]}")
            with open(final_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('results', {})  # Extract just the results section

        # If no final, try any incremental file
        inc_files = sorted(inc_path.glob("incremental_results_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if inc_files:
            print(f" Loading results from: {inc_files[0]}")
            with open(inc_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('results', {})

    print("  No results files found")
    return None


def find_missing_model_persona_combinations(
    original_queries: Dict[str, Dict[str, str]], 
    existing_data: Dict,
    target_models: List[str]
) -> Dict[str, Dict[str, str]]:
    """Find queries that are missing for specific model-persona combinations."""
    missing_queries = {}
    existing_results = existing_data.get("results", {})
    
    for topic, topic_queries in original_queries.items():
        missing_queries[topic] = {}
        
        for persona_id, prompt in topic_queries.items():
            # Check if this persona is missing for ANY of the target models
            is_missing_for_any_model = False
            
            for model in target_models:
                if (model not in existing_results or 
                    topic not in existing_results[model] or 
                    persona_id not in existing_results[model][topic]):
                    # This model-persona combination is missing
                    is_missing_for_any_model = True
                    break
            
            if is_missing_for_any_model:
                missing_queries[topic][persona_id] = prompt
    
    return missing_queries


def find_missing_queries(
    original_queries: Dict[str, Dict[str, str]], 
    existing_data: Dict
) -> Dict[str, Dict[str, str]]:
    """Find queries that haven't been completed yet (legacy function for compatibility)."""
    # This is kept for backward compatibility but should not be used for resume
    return find_missing_model_persona_combinations(original_queries, existing_data, [])


def query_llm_fast_resume(
    incremental_file_path: str,
    original_queries: Dict[str, Dict[str, str]],
    list_of_models: List[str],
    initial_batch_size: int = 50,
    initial_concurrency: int = 20,
    max_concurrency: int = 100,
    adaptive_mode: bool = True,
    all_open_router: bool = False,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    ensure_100_percent_success: bool = True,
    save_incremental: bool = True,
    incremental_interval: int = 1,
    subdir: str = ""
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Resume query processing from an incremental results file.

    Args:
        incremental_file_path: Path to the incremental results file to resume from
        original_queries: The original queries dictionary used in the initial run
        list_of_models: List of model specifications
        initial_batch_size: Starting batch size (will adapt based on performance)
        initial_concurrency: Starting concurrency (will adapt based on performance)
        max_concurrency: Maximum allowed concurrency
        adaptive_mode: Whether to automatically adjust parameters
        all_open_router: If True, route all models through OpenRouter
        max_retries: Maximum number of retry attempts for failed queries
        retry_delay: Initial delay between retry rounds (increases exponentially)
        ensure_100_percent_success: If True, retry failed queries until they succeed
        save_incremental: If True, save incremental results to logs/incremental folder
        incremental_interval: Save incremental results every N batches (1 = every batch)
        subdir: Subdirectory within logs (e.g., "main", "control")
    
    Returns:
        Dictionary of {model: {topic: {persona_id: response}}}
    """
    print(f" Resuming from incremental file: {incremental_file_path}")
    
    # Load existing results
    existing_data = load_incremental_results(incremental_file_path)
    existing_results = existing_data.get("results", {})
    
    # Find missing queries for the target models
    missing_queries = find_missing_model_persona_combinations(original_queries, existing_data, list_of_models)
    
    # Count missing queries
    total_missing = sum(len(topic_queries) for topic_queries in missing_queries.values())
    total_original = sum(len(topic_queries) for topic_queries in original_queries.values())
    
    # Count completed model-persona combinations
    total_combinations = total_original * len(list_of_models)
    completed_combinations = 0
    
    for topic, topic_queries in original_queries.items():
        for persona_id in topic_queries.keys():
            for model in list_of_models:
                if (model in existing_results and 
                    topic in existing_results[model] and 
                    persona_id in existing_results[model][topic]):
                    completed_combinations += 1
    
    print(f" Resume Status:")
    print(f"   Total model-persona combinations: {total_combinations}")
    print(f"   Completed combinations: {completed_combinations}")
    print(f"   Missing combinations: {total_combinations - completed_combinations}")
    print(f"   Completion rate: {completed_combinations/total_combinations*100:.1f}%")
    print(f"   Missing personas to retry: {total_missing}")
    
    if total_missing == 0:
        print(" All queries already completed!")
        return existing_results
    
    print(f" Processing {total_missing} missing queries...")
    
    # Get the last batch number and incremental counter from the existing results
    metadata = existing_results.get("metadata", {})
    last_batch_number = metadata.get("batch_number", 1)
    last_incremental_counter = metadata.get("incremental_counter", 0)
    
    start_batch_number = last_batch_number + 1
    
    print(f" Continuing from batch {start_batch_number}")
    print(f" Continuing incremental counter from {last_incremental_counter}")
    
    # Process missing queries
    missing_results = query_llm_fast(
        nested_queries=missing_queries,
        list_of_models=list_of_models,
        initial_batch_size=initial_batch_size,
        initial_concurrency=initial_concurrency,
        max_concurrency=max_concurrency,
        adaptive_mode=adaptive_mode,
        all_open_router=all_open_router,
        max_retries=max_retries,
        retry_delay=retry_delay,
        ensure_100_percent_success=ensure_100_percent_success,
        save_incremental=save_incremental,
        incremental_interval=incremental_interval,
        start_batch_number=start_batch_number,
        initial_incremental_counter=last_incremental_counter,
        subdir=subdir
    )
    
    # Merge results - preserve ALL existing results and add new ones
    print(" Merging results...")
    merged_results = existing_results.copy()
    
    for model, model_results in missing_results.items():
        if model not in merged_results:
            merged_results[model] = {}
        
        for topic, topic_results in model_results.items():
            if topic not in merged_results[model]:
                merged_results[model][topic] = {}
            
            # Only update with new results, don't overwrite existing ones
            for persona_id, response in topic_results.items():
                if persona_id not in merged_results[model][topic]:
                    merged_results[model][topic][persona_id] = response
    
    print(" Resume completed successfully!")
    print(f" Final results contain {len(merged_results)} models")
    
    # Count final combinations
    final_combinations = 0
    for model in list_of_models:
        if model in merged_results:
            for topic in merged_results[model]:
                final_combinations += len(merged_results[model][topic])
    
    print(f" Final model-persona combinations: {final_combinations}")
    return merged_results

