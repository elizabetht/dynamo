<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## DeepSeek-V4 assistant-prefix continuation

For the SGLang custom DeepSeek-V4 encoder, an explicit
`continue_final_message: true` now removes the final assistant message before
rendering the conversation and appends its tokenized text prefix afterward,
matching native SGLang. Previously the prefix was rendered as a completed message
ending in EOS; thinking mode also inserted a closing reasoning marker before it.
The prefix tokenizer's leading BOS, if present, is removed. Ordinary completed
messages and requests ending in a user message retain their existing behavior.
This change concerns the custom encoder's top-level continuation option; it does
not change Hugging Face template option handling.

CPU validation used DeepSeek-V4-Flash tokenizer revision
`60d8d70770c6776ff598c94bb586a859a38244f1` and SGLang v0.5.19 commit
`0bcd822377da7b5718e674eaf9c870d349424dd1` (encoder and serving code
verified by SHA256 against that commit).
The regression suite produced six baseline failures and two passing controls;
the candidate passed all 15 focused DeepSeek tests and 299 available processor
CPU tests (one uncached byte-fallback tokenizer test was deselected).
The unit suite uses its existing small Qwen tokenizer fixture; the HTTP replay
uses the pinned DeepSeek tokenizer. Native HTTP validation reused bindings built
from `70e094f650ab89eb81d359825adcb78fc9256aba`; it is not a full rebuild
of this Python candidate.
Four local native HTTP/TCP cases (streaming/non-streaming, chat/thinking) delivered
incorrect prompt tokens on the baseline and matching prefix tokens on the
candidate. The worker returns synthetic tokens: this proves preprocessing and
transport behavior, not model generation or schema enforcement. Cluster validation
is pending. No performance improvement has been measured.

Run the focused CPU regressions in an environment with Dynamo bindings and SGLang:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k deepseek_v4 -q
```

For the native HTTP reproduction, download only `config.json`, `tokenizer.json`
and `tokenizer_config.json` from the pinned tokenizer revision into a directory,
then run:

```bash
PYTHONPATH=components/src python \
  components/src/dynamo/frontend/tests/reproduce_dsv4_continuation.py \
  --model-path /path/to/pinned-tokenizer --output /tmp/dsv4-continuation
```

The same script with `--expect-baseline` asserts the four baseline mismatches.
It creates its own local request namespace and dynamic port and loads no weights.
Full native runtime shutdown remains unqualified: the local build reported nine
remaining tasks at interpreter exit on both revisions. Real generation and
resource cleanup require separate cluster validation.
