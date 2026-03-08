"""
Unified LLM client supporting Anthropic, Qwen (via OpenAI-compat), and DeepSeek.

Usage:
    client = LLMClient()          # uses LLM_PROVIDER from config
    client = LLMClient("qwen")    # explicit provider

    response = client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a helpful assistant.",
        model=get_model("searcher"),
        temperature=0.0,
    )
    print(response.content)  # str

    # With tool use (Anthropic / Qwen only):
    response = client.chat(..., tools=[...])
    for block in response.tool_calls:
        print(block.name, block.input)
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any

import config as cfg


# ---------------------------------------------------------------------------
# Unified response types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ChatResponse:
    content: str                    # text content (may be empty if only tool calls)
    tool_calls: list[ToolCall]      # empty list if no tools called
    stop_reason: str                # "end_turn" | "tool_use" | "stop" | "length"
    raw: Any                        # original SDK response object


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified client for Anthropic, Qwen, and DeepSeek.

    Tool use behaviour:
    - Anthropic: native tool_use via messages API
    - Qwen:      native tool_use via OpenAI-compat API
    - DeepSeek:  JSON-in-text fallback (no native tool_use support in reasoner)
    """

    def __init__(self, provider: str | None = None):
        self.provider = provider or cfg.LLM_PROVIDER
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        elif self.provider == "qwen":
            from openai import OpenAI
            return OpenAI(
                api_key=cfg.DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif self.provider == "deepseek":
            from openai import OpenAI
            return OpenAI(
                api_key=cfg.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com",
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}. "
                             "Choose 'anthropic', 'qwen', or 'deepseek'.")

    def supports_tool_use(self) -> bool:
        return self.provider in ("anthropic", "qwen")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """
        Send a chat request and return a unified ChatResponse.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
                         For tool results use the provider-native format;
                         this method handles normalisation internally.
            system:      System prompt string.
            model:       Model name. Defaults to provider's searcher model.
            tools:       Anthropic-format tool definitions (list of dicts with
                         name, description, input_schema). Converted to OpenAI
                         format automatically for Qwen/DeepSeek.
            temperature: Sampling temperature.
            max_tokens:  Max output tokens.
        """
        model = model or cfg.get_model("searcher")

        if self.provider == "anthropic":
            return self._chat_anthropic(messages, system, model, tools, temperature, max_tokens)
        else:
            return self._chat_openai(messages, system, model, tools, temperature, max_tokens)

    # ------------------------------------------------------------------
    # Anthropic implementation
    # ------------------------------------------------------------------

    def _chat_anthropic(self, messages, system, model, tools, temperature, max_tokens):
        import anthropic

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools  # already in Anthropic format

        raw = self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        stop = raw.stop_reason or "end_turn"
        return ChatResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # OpenAI-compat implementation (Qwen / DeepSeek)
    # ------------------------------------------------------------------

    def _chat_openai(self, messages, system, model, tools, temperature, max_tokens):
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(self._normalise_messages_for_openai(messages))

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        oai_tools = None
        if tools and self.supports_tool_use():
            oai_tools = [self._anthropic_tool_to_openai(t) for t in tools]
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"

        raw = self._client.chat.completions.create(**kwargs)
        choice = raw.choices[0]
        msg = choice.message

        text_content = msg.content or ""
        tool_calls = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        elif not self.supports_tool_use() and tools:
            # JSON-in-text fallback for DeepSeek
            tool_calls = self._extract_tool_calls_from_text(text_content)

        finish = choice.finish_reason or "stop"
        stop_reason = "tool_use" if tool_calls else ("end_turn" if finish in ("stop", "eos") else finish)

        return ChatResponse(
            content=text_content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Format converters
    # ------------------------------------------------------------------

    @staticmethod
    def _anthropic_tool_to_openai(tool: dict) -> dict:
        """Convert Anthropic tool definition to OpenAI function tool format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _normalise_messages_for_openai(messages: list[dict]) -> list[dict]:
        """
        Convert Anthropic-style tool_result content blocks into OpenAI tool messages.
        Pass through all other messages unchanged.
        """
        result = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                # May contain tool_result blocks
                tool_results = []
                text_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_results.append(block)
                    else:
                        text_parts.append(block)

                for tr in tool_results:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr.get("content", ""),
                    })
                if text_parts:
                    result.append({"role": "user", "content": text_parts})
            elif msg["role"] == "assistant" and isinstance(msg["content"], list):
                # Convert assistant tool_use blocks to tool_calls format
                text_parts = []
                tool_calls = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)

                oai_msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)
            else:
                result.append(msg)
        return result

    @staticmethod
    def _extract_tool_calls_from_text(text: str) -> list[ToolCall]:
        """
        JSON-in-text fallback for providers without native tool use.
        Looks for patterns like:
          {"tool": "tool_name", "args": {...}}
        or
          TOOL_CALL: {"tool": "tool_name", "args": {...}}
        """
        tool_calls = []
        # Match JSON objects that have a "tool" key
        pattern = r'\{[^{}]*"tool"\s*:\s*"[^"]+[^{}]*\}'
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                obj = json.loads(match.group())
                if "tool" in obj:
                    tool_calls.append(ToolCall(
                        id=f"fallback_{len(tool_calls)}",
                        name=obj["tool"],
                        input=obj.get("args", obj.get("input", {})),
                    ))
            except json.JSONDecodeError:
                pass
        return tool_calls
