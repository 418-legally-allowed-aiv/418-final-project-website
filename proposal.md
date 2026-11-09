
---
layout: default
title: Project Proposal
---

# 418 Final Project Proposal

**Sarah Huang (sarahh2), Gene Yang (geney)**

**November 9, 2026**

[← Back to Homepage](./)

---

## Summary

We will implement and evaluate a chunked algorithm for Recirculation, a new LLM inference method that introduces heavy sequentialness. We will profile the performance and optimize it against the GPU hardware we use, end up with a performant optimized system for running it, and benchmark it along speed and prediction quality against vanilla inference and recirculation.

## Background

Traditional transformers process all the tokens in an input prompt in parallel. This is possible because, although each token attends to previous tokens, the computation at each layer can be done across all token positions at once. GPUs are particularly efficient at this process (prefill) because of the large amount of parallel computation.

Recirculation is a recently proposed technique proposed to remedy a transformer's state tracking ability. It does this by taking the hidden state from a later layer, linearly mixing it into an earlier layer at the same token position, and recomputing the subsequent layers. The updated information is stored in the KV cache, allowing later tokens to use it. However, this introduces dependencies between consecutive tokens, making prefill largely sequential. Existing implementations can partially address this by overlapping computation across tokens, but the sequential dependency remains.

Instead of processing each token individually, we propose processing a chunk of tokens in parallel, then recirculating the entire chunk before moving to the next chunk. This means that tokens within a chunk cannot fully benefit from the recirculation of earlier tokens in that chunk, making the computation approximate. We aim to investigate whether the we can optimize this parallelism, and whether it can significantly accelerate prefill without losing much of Recirculation's improvement in model quality.

## The Challenge

### Difficulties

1. The most obvious challenge is in parallelizing the work itself. We (and the papers' authors) hypothesize that speedup should be possible by moving the sequentialness up from tokens to chunks, but it has not been implemented yet by anyone.

2. Given finite computational resources, how should one allocate the different dimensions of parallelism? This requires careful profiling and reasoning about the interplay between existing data and tensor parallelism and our parallelism over tokens and layers.

3. Managing the KV cache(s) (the original and recirculated copies) will require reasoning about memory and bandwidth, as well as communication.

4. There are a lot of moving parts with a machine learning research codebase, so profiling will have to be done extremely rigorously to be useful.

### Workload

For regular prefill, all transformer layers can be computed on all tokens in parallel. With recirculation, only earlier transformer layers can be computed over every token in parallel, while later transformer layers depend on the corrected post-recirculation KV entries from previous tokens. For both, there are sequential dependencies between layers, but for recirculation there is opportunity for overlapping computation.

Our proposed solution involves repeated attention over an increasingly large KV cache, increasing in size by chunks. Recirculation also requires replaying upper layers and updating their cached representations, introducing additional computation and memory traffic.

### Constraints

GPUs need a high parallel workload, and there is unavoidable sequentialness and synchronization in our workload. GPU memory hierarchy also determins a lot of the performance, and the memory management involved in recirculation is quite complicated. The physical constraints of whichever hardware we work on will determine a lot of our design choices.

## Resources

1. [Recirculation GitHub Repository](https://github.com/ModelCloud/Recirculation)
2. [Recirculation Paper](https://arxiv.org/pdf/2608.17981v3)
3. Booksum/chapter dataset
4. We need a GPU that can serve as a platform for Gemma 3 1B, so potentially 24GB RTX 4090.
5. PyTorch profiler

## Goals and Deliverables

### Plan to Achieve

We hope to implement a function version of chunked recirculation such that when it is run with K = 1 where K is the number of tokens per chunk, its performance closely follows the sequential recirculation. The prefill latency and memory usage will be measured for at least 4 different chunk sizes against the existing sequential baseline, providing a proof of concept. We will also evaluate the perplexity of such chunks and compare them against the baseline sequential implementation described in the paper. With this analysis, we will provide a deliverable that analyzes and explains the speedup and its limitations from the profiling done, as well as a synchronization analysis done similar to those done in previous assignments.

### Hope to Achieve

We hope to achieve at least 2× the prefill speedup with minimal increase in perplexity on the Booksum/Chapter dataset with a specific configuration of chunks. With the remaining time, we would want to evaluate the effects of chunking and, if the initial implementation provided optimistic speedup numbers, investigate further optimizations for this specific algorithm.

If the speedup on the initial chunking implementation is not significant, or comes at a high cost of perplexity, we can conclude the project by clearly illustrating the bottlenecks of this algorithm, and proposing possible alternatives.

## Platform Choice

We wish to use NVIDIA GPUs alongside PyTorch as our main hardware and software. PyTorch is fine-grained enough so that we will be able to have explicit GPU scheduling and memory management, isolating the key parts of the algorithm, while learning on pre-existing software and kernels for the ML part of things (matrix multiplications, activation functions, attention).

We initially plan to target a single GPU, since the main parallelism we want to exploit is across tokens and transformer layers within one inference request. This avoids introducing inter-GPU communication costs that are not central to our research question.

## Schedule

1. **Week 1 (Nov 9–15):** Set up environments and platforms needed, run and profile existing recirculation implementation, and inspect the related prefill logic to form initial ideas of parallelization.

2. **Week 2 (Nov 16–22):** Implement chunked prefill, at least have K = 1 with the exact same perplexity as the initial recirculation implementation.

3. **Week 3 (Nov 23–29):** Finish implementation for K > 1, submit milestone report (NOV 23), test multiple chunk sizes for correctness. Set up evaluation for speedup and perplexity.

4. **Week 4 (Nov 30–Dec 6):** Run and profile new chunked implementation. If enough time, investigate a further optimization, or alternative approach, depending on the outcome of the proof of concept.

5. **Week 5 (Dec 7–10):** Finish writeup and figures, create poster, and practice presentation.

6. **Final Presentation (Dec 11, 5:30–8:30 PM):** Print out poster and do last run through of presentation.

---

[← Back to Homepage](./)
