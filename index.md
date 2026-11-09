
# Chunked Recirculation for Faster LLM Inference

**CMU 15-418/618 — Fall 2026**

**Team Members:** Sarah Huang (sarahh2), Gene Yang (geney)

## Project Overview

We are investigating chunked recirculation, an approach to improving the parallelism of Recirculation-based LLM inference.

Traditional transformer prefill processes tokens in parallel, while Recirculation introduces sequential dependencies between tokens. Our project explores whether processing groups of tokens in chunks can improve prefill performance while preserving model quality.

We will implement and benchmark chunked recirculation on NVIDIA GPUs using PyTorch, evaluating prefill latency, memory usage, and perplexity.

## Project Documents

- [Project Proposal](proposal.html)
- Milestone Report (Coming November 23)
- Final Report (Coming December 10)

## Project Timeline

| Week | Dates | Goals |
|---|---|---|
| 1 | Nov 9–15 | Environment setup, baseline profiling |
| 2 | Nov 16–22 | Implement chunked prefill with K = 1 |
| 3 | Nov 23–29 | Support K > 1, milestone report, initial evaluation |
| 4 | Nov 30–Dec 6 | Benchmark and profile chunked implementation |
| 5 | Dec 7–10 | Final report, figures, poster preparation |
| Presentation | Dec 11 | Final poster session |

## Resources

- [Recirculation Implementation](https://github.com/ModelCloud/Recirculation)
- [Recirculation Paper](https://arxiv.org/pdf/2608.17981v3)
