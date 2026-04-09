"""
LeanAI — Streaming Response Handler
Shows tokens appearing in real-time instead of waiting for full generation.

Transforms the experience from:
  "stare at blank screen for 90 seconds, then wall of text"
to:
  "words appear immediately, read as they generate"

Works with both CLI and API/WebSocket.
"""

import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Generator


@dataclass
class StreamConfig:
    """Configuration for streaming output."""
    enabled: bool = True
    flush_every: int = 1       # flush after every N tokens
    show_cursor: bool = True   # show blinking cursor while generating
    color: bool = True         # use ANSI colors


class StreamingGenerator:
    """
    Wraps llama.cpp model to stream tokens to the terminal in real-time.
    
    Usage:
        streamer = StreamingGenerator(model, prompt_format="chatml")
        
        # Stream to terminal
        full_text = streamer.generate_streaming(
            system="You are helpful.",
            user="Explain quicksort",
        )
        # Tokens appear word by word as they generate
    """

    def __init__(self, model=None, prompt_format: str = "chatml",
                 config: Optional[StreamConfig] = None):
        self.model = model
        self.prompt_format = prompt_format
        self.config = config or StreamConfig()

    def _build_prompt(self, system: str, user: str) -> tuple:
        """Build prompt and stop tokens based on format."""
        if self.prompt_format == "chatml":
            prompt = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            stop = ["<|im_end|>", "<|im_start|>"]
        else:
            prompt = (
                f"<|system|>\n{system}<|end|>\n"
                f"<|user|>\n{user}<|end|>\n"
                f"<|assistant|>\n"
            )
            stop = ["<|end|>", "<|user|>", "<|assistant|>"]
        return prompt, stop

    def generate_streaming(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Generate response with real-time streaming to terminal.
        
        Args:
            system: system prompt
            user: user prompt
            max_tokens: max tokens to generate
            temperature: sampling temperature
            callback: optional function(token_text) called for each token
        
        Returns:
            Complete generated text
        """
        if not self.model:
            return ""

        prompt, stop = self._build_prompt(system, user)

        full_text = ""
        token_count = 0
        start_time = time.time()

        try:
            # Use llama.cpp streaming
            for chunk in self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
            ):
                token_text = chunk["choices"][0]["text"]
                full_text += token_text
                token_count += 1

                # Output to terminal
                if self.config.enabled:
                    sys.stdout.write(token_text)
                    if token_count % self.config.flush_every == 0:
                        sys.stdout.flush()

                # Callback for API/WebSocket
                if callback:
                    callback(token_text)

        except Exception as e:
            if self.config.enabled:
                sys.stdout.write(f"\n[Stream error: {e}]")
            full_text += f"\n[Generation interrupted: {e}]"

        # Final flush
        if self.config.enabled:
            sys.stdout.flush()

        return full_text.strip()

    def generate_non_streaming(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Non-streaming fallback — standard generation."""
        if not self.model:
            return ""

        prompt, stop = self._build_prompt(system, user)

        result = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return result["choices"][0]["text"].strip()

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        stream: bool = True,
        callback: Optional[Callable] = None,
    ) -> str:
        """Generate with automatic streaming/non-streaming based on config."""
        if stream and self.config.enabled:
            return self.generate_streaming(
                system, user, max_tokens, temperature, callback
            )
        else:
            return self.generate_non_streaming(
                system, user, max_tokens, temperature
            )


def print_streaming_header():
    """Print a visual header before streaming starts."""
    sys.stdout.write("\nLeanAI:\n")
    sys.stdout.flush()


def print_streaming_footer(latency_ms: float, confidence: float, tier: str,
                           from_mem: bool = False, verified: bool = False,
                           cached: bool = False):
    """Print metadata after streaming completes."""
    conf_label = "High" if confidence >= 90 else "Good" if confidence >= 70 else "Moderate" if confidence >= 50 else "Low"
    conf_bars = int(confidence / 5)
    conf_display = "█" * conf_bars + "░" * (20 - conf_bars)

    print(f"\n───────────────────────────────────────────────────────")
    meta = f"Confidence  [{conf_display}] {confidence:.0f}%  {conf_label}"
    meta += f"\nTier: {tier}  Latency: {latency_ms:.0f}ms"
    if from_mem:
        meta += "  From memory: yes"
    if verified:
        meta += "  Verified: yes"
    if cached:
        meta += "  ⚡ CACHED"
    print(meta)
