"""Compatibility parser for the deployed MiniMax-M3 tool-call format.

MiniMax-M3 checkpoints can omit the opening tag of an invoke's first
parameter while still emitting the matching closing tag.  vLLM's built-in
Rust parser rejects that form (vllm-project/vllm#51073).  Repair only those
elided parameter tags, then delegate schema coercion and OpenAI response
construction to vLLM's built-in parser.

The accuracy adapters in this directory use non-streaming chat completions.
Streaming deliberately remains on the upstream parser path.
"""

import re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ExtractedToolCallInformation
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.minimax_m3_tool_parser import MinimaxM3ToolParser

_NAMESPACE = "]<]minimax[>["
_TOOL_CALL_START = f"{_NAMESPACE}<tool_call>"
_TOOL_CALL_END = f"{_NAMESPACE}</tool_call>"
_TOOL_CALL_BODY = re.compile(
    rf"({re.escape(_TOOL_CALL_START)})(.*?)(?={re.escape(_TOOL_CALL_END)}|\Z)",
    re.DOTALL,
)
_ELIDED_PARAMETER = re.compile(
    rf"{re.escape(_NAMESPACE)}(?!<)(.*?){re.escape(_NAMESPACE)}"
    r"</([A-Za-z_][A-Za-z0-9_.:-]*)>",
    re.DOTALL,
)


def repair_elided_parameter_tags(model_output: str) -> str:
    """Insert the opening parameter tag named by an emitted closing tag."""

    def repair_body(match: re.Match[str]) -> str:
        def repair_parameter(parameter: re.Match[str]) -> str:
            value, name = parameter.groups()
            return (
                f"{_NAMESPACE}<{name}>{value}"
                f"{_NAMESPACE}</{name}>"
            )

        return match.group(1) + _ELIDED_PARAMETER.sub(
            repair_parameter, match.group(2)
        )

    return _TOOL_CALL_BODY.sub(repair_body, model_output)


@ToolParserManager.register_module("minimax_m3_tolerant")
class TolerantMinimaxM3ToolParser(MinimaxM3ToolParser):
    """Use upstream MiniMax-M3 parsing after repairing its known wire variant."""

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Keep MiniMax namespace/tool tags available to the parser.

        The upstream Rust MiniMax-M3 parser does not request preservation of
        special tokens.  Some deployed M3 tokenizers classify the namespace
        and tool wrapper as special, so the default decode path removes the
        very delimiters needed by both the upstream and tolerant grammars.
        """
        request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        return super().extract_tool_calls(
            repair_elided_parameter_tags(model_output), request
        )
