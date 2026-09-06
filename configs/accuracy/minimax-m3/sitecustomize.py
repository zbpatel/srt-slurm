"""Register the MiniMax-M3 compatibility parser in Dynamo frontends.

Dynamo's vLLM chat processor looks up tool parsers directly through vLLM's
``ToolParserManager`` and does not process vLLM's ``--tool-parser-plugin``
flag.  This module is loaded by Python during frontend startup when this
directory is on ``PYTHONPATH``, before Dynamo constructs its chat processor.
"""

import minimax_m3_tolerant_tool_parser  # noqa: F401
