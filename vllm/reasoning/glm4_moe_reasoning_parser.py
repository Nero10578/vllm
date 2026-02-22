# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

logger = init_logger(__name__)


class IdentityReasoningParser(ReasoningParser):
    """
    Identity reasoning parser.

    This parser does not attempt to parse or strip out reasoning tokens.
    It treats the entire model output as content and ignores reasoning.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # Always return True, since we never treat reasoning specially
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Identity: return all tokens as content
        return input_ids

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        # Just wrap delta_text as content, ignore reasoning
        if delta_text:
            return DeltaMessage(content=delta_text)
        return None

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        # No reasoning separation: return None for reasoning,
        # and full model_output as content
        return None, model_output


@ReasoningParserManager.register_module("glm45")
class Glm4MoeModelReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Glm4MoeModel model.
    
    Delegates to either DeepSeekR1ReasoningParser or IdentityReasoningParser
    based on `thinking` and `enable_thinking`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if thinking is None and enable_thinking is None:
            chat_kwargs["thinking"] = True
            chat_kwargs["enable_thinking"] = True
            kwargs["chat_template_kwargs"] = chat_kwargs
            
        super().__init__(tokenizer, *args, **kwargs)

        thinking = bool(chat_kwargs.get("thinking", False))
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        thinking = thinking or enable_thinking

        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        return self._parser.extract_reasoning_content(model_output, request)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        return self._parser.extract_reasoning_content_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
