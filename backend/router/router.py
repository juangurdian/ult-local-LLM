from typing import Optional, Dict, Any, List
import asyncio
import time
from datetime import datetime

from .classifier import QueryClassifier, TaskType
import ollama


class ModelRouter:
    """Intelligent model router that automatically selects optimal models."""

    def __init__(self):
        self.classifier = QueryClassifier()
        self.client = ollama.Client()

        # Model capabilities and configurations
        self.model_configs = {
            "qwen3:4b": {
                "context_window": 4096,
                "strengths": ["speed", "simple_chat"],
                "weaknesses": ["complex_reasoning", "long_context"],
                "estimated_tokens_per_sec": 50,
                "estimated_vram_gb": 4,
            },
            "qwen3:8b": {
                "context_window": 8192,
                "strengths": ["balanced", "general_purpose"],
                "weaknesses": ["very_complex_tasks"],
                "estimated_tokens_per_sec": 35,
                "estimated_vram_gb": 6,
            },
            "deepseek-r1:8b": {
                "context_window": 16384,
                "strengths": ["reasoning", "analysis", "complex_tasks"],
                "weaknesses": ["speed"],
                "estimated_tokens_per_sec": 25,
                "estimated_vram_gb": 6,
            },
            "qwen2.5-coder:7b": {
                "context_window": 8192,
                "strengths": ["coding", "programming", "debugging"],
                "weaknesses": ["general_chat"],
                "estimated_tokens_per_sec": 35,
                "estimated_vram_gb": 6,
            },
            "llava:7b": {
                "context_window": 4096,
                "strengths": ["vision", "image_analysis"],
                "weaknesses": ["text_only_tasks"],
                "estimated_tokens_per_sec": 25,
                "estimated_vram_gb": 6,
            }
        }

    async def route_query(
        self,
        query: str,
        images: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        force_model: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Route a query to the optimal model."""

        start_time = time.time()

        # Allow manual model override
        if force_model and force_model != "auto":
            if force_model not in self.model_configs:
                available_models = list(self.model_configs.keys())
                raise ValueError(f"Unknown model: {force_model}. Available: {available_models}")

            return {
                "model": force_model,
                "routing_method": "manual_override",
                "reasoning": f"User specified model: {force_model}",
                "confidence": 1.0,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": datetime.now().isoformat()
            }

        # Analyze the query
        has_images = images is not None and len(images) > 0
        context_length = len(context) if context else 0

        classification = self.classifier.classify(
            query=query,
            has_images=has_images,
            context_length=context_length,
            conversation_history=conversation_history
        )

        # Get model recommendation
        recommendation = self.classifier.get_model_recommendation(classification)

        # Validate context length doesn't exceed model limits
        estimated_tokens = self._estimate_token_count(query, context)
        selected_model = recommendation["model"]

        if estimated_tokens > self.model_configs[selected_model]["context_window"] * 0.8:
            # If context is too long, prefer models with larger context windows
            if estimated_tokens <= self.model_configs["deepseek-r1:8b"]["context_window"]:
                selected_model = "deepseek-r1:8b"
                recommendation["reason"] += " (upgraded for context length)"
            else:
                recommendation["warning"] = "Query may exceed context window limits"

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "model": selected_model,
            "routing_method": "intelligent",
            "task_type": recommendation["task_type"],
            "confidence": recommendation["confidence"],
            "complexity": recommendation["complexity"],
            "reasoning": recommendation["reason"],
            "estimated_tokens": estimated_tokens,
            "expected_speed": recommendation.get("expected_speed", "Unknown"),
            "keywords_found": [str(kw) for kw in recommendation.get("keywords", [])],
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat(),
            "classification_details": {
                "task_type": classification.task_type.value,
                "confidence": classification.confidence,
                "complexity_score": classification.complexity_score,
                "reasoning": classification.reasoning
            }
        }

    async def execute_query(
        self,
        routing_result: Dict[str, Any],
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the routed query and return the response."""

        model = routing_result["model"]
        start_time = time.time()

        # Pack context respecting the model context window (and include optional summary)
        packed_messages, packing_stats = self._pack_messages(messages, model)

        try:
            # Prepare the request for Ollama
            ollama_messages = []
            for msg in packed_messages:
                ollama_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }

                # Handle images for vision models
                if msg.get("images") and routing_result.get("task_type") == "vision":
                    # Convert base64 images to proper format if needed
                    ollama_msg["images"] = msg["images"]

                ollama_messages.append(ollama_msg)

            # Make the request
            response = self.client.chat(
                model=model,
                messages=ollama_messages,
                stream=False,
                options={
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 2048)
                }
            )

            execution_time = int((time.time() - start_time) * 1000)
            response_content = response.get("message", {}).get("content", "")

            return {
                "success": True,
                "response": response_content,
                "model_used": model,
                "execution_time_ms": execution_time,
                "tokens_generated": self._estimate_token_count(response_content),
                "routing_info": routing_result,
                "packing": packing_stats,
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger = logging.getLogger(__name__)
            logger.error(
                "Execution error for model=%s: %s",
                model,
                str(e),
            )
            return {
                "success": False,
                "error": str(e),
                "model_used": model,
                "execution_time_ms": execution_time,
                "routing_info": routing_result,
                "packing": packing_stats,
            }

    def get_available_models(self) -> Dict[str, Any]:
        """Get information about all available models."""
        return {
            model_name: {
                **config,
                "model_name": model_name,
                "status": "available"  # Could check actual availability
            }
            for model_name, config in self.model_configs.items()
        }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics (placeholder for future implementation)."""
        return {
            "total_queries": 0,
            "model_usage": {model: 0 for model in self.model_configs.keys()},
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0
        }

    def _estimate_token_count(self, text: str, context: Optional[str] = None) -> int:
        """Roughly estimate token count for a text string."""
        if not text:
            return 0

        # Simple estimation: ~4 characters per token for English text
        # This is a rough approximation - real tokenization is more complex
        total_chars = len(text)
        if context:
            total_chars += len(context)

        estimated_tokens = total_chars // 4

        # Add some overhead for special tokens and formatting
        return int(estimated_tokens * 1.1)

    def _pack_messages(self, messages: List[Dict[str, Any]], model: str):
        """
        Trim the conversation history to fit within the model context window.
        Strategy: keep most recent messages until ~80% of the context window is used.
        Adds a lightweight rolling summary of dropped history.
        """
        context_window = self.model_configs.get(model, {}).get("context_window", 4096)
        target_tokens = int(context_window * 0.8)

        kept = []
        total_tokens = 0
        dropped: List[Dict[str, Any]] = []

        # Walk from newest to oldest, accumulate until the limit
        for msg in reversed(messages):
            msg_tokens = self._estimate_token_count(msg.get("content", ""))
            if total_tokens + msg_tokens <= target_tokens or not kept:
                kept.append(msg)
                total_tokens += msg_tokens
            else:
                dropped.append(msg)

        kept.reverse()
        dropped.reverse()

        tokens_kept = total_tokens
        tokens_total = sum(self._estimate_token_count(m.get("content", "")) for m in messages)
        tokens_dropped = max(tokens_total - tokens_kept, 0)

        used_summary = False
        summary_tokens = 0

        # Create a lightweight summary of dropped content if any
        summary_msg = None
        if dropped:
            used_summary = True
            summary_text = self._build_summary(dropped)
            summary_tokens = self._estimate_token_count(summary_text)
            # Only prepend summary if it fits comfortably
            if summary_tokens + tokens_kept < target_tokens:
                summary_msg = {
                    "role": "system",
                    "content": f"Conversation summary so far: {summary_text}",
                }
                kept.insert(0, summary_msg)
                tokens_kept += summary_tokens

        packing_stats = {
            "context_window": context_window,
            "target_tokens": target_tokens,
            "tokens_total": tokens_total,
            "tokens_kept": tokens_kept,
            "tokens_dropped": tokens_dropped,
            "used_summary": used_summary,
            "summary_tokens": summary_tokens,
        }

        return kept, packing_stats

    def _build_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Lightweight deterministic summary (no extra model call)."""
        # Take up to the last 3 dropped user/assistant turns and compress them.
        snippets = []
        for msg in messages[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            snippets.append(f"{role}: {content}")
        joined = " | ".join(snippets)
        # Truncate to keep tokens small
        return joined[:600]
