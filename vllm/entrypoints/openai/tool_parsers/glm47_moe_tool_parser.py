# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.entrypoints.openai.tool_parsers.glm4_moe_tool_parser import (
    Glm4MoeModelToolParser,
)

logger = init_logger(__name__)


@ToolParserManager.register_module("glm47")
class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    """
    Tool parser for GLM 4.7 model.
    
    Extends GLM 4.5 parser with updated regex patterns to handle
    the slightly different tool call format used by GLM 4.7.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )