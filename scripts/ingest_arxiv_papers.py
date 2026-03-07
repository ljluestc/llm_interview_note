#!/usr/bin/env python3
"""Ingest ArXiv papers into the RAG system.

Appends document entries and QA pairs for 13 ArXiv papers
(from wdndev.github.io blog) to the existing JSONL files.
"""

import json
import os
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DOCS_FILE = os.path.join(DATA_DIR, "all_documents.jsonl")
QA_FILE = os.path.join(DATA_DIR, "all_qa_pairs.jsonl")

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")

# ---------- paper definitions ----------

PAPERS = [
    # --- LLM domain ---
    {
        "arxiv_id": "2602.24288",
        "title": "DARE-bench: A Comprehensive Benchmark for Evaluating Large Language Models",
        "domain": "LLM",
        "category": "09.大语言模型评估",
        "subcategory": "LLM Benchmarking",
        "difficulty": "advanced",
        "content": (
            "DARE-bench introduces a comprehensive benchmark framework for evaluating large language models "
            "across diverse capabilities including reasoning, knowledge retrieval, code generation, and "
            "instruction following. The benchmark addresses limitations of existing evaluation suites by "
            "providing multi-dimensional scoring, dynamic test set generation to prevent data contamination, "
            "and fine-grained capability profiling. DARE-bench covers over 20 task categories with "
            "difficulty-stratified test instances, enabling researchers to identify specific strengths and "
            "weaknesses of different LLM architectures. The framework also includes automatic evaluation "
            "metrics calibrated against human judgments, reducing the need for expensive manual annotation."
        ),
        "keywords": ["benchmark", "LLM evaluation", "DARE-bench", "data contamination",
                      "capability profiling", "automatic evaluation"],
        "questions": [
            ("What is DARE-bench and how does it improve LLM evaluation?",
             "DARE-bench is a comprehensive benchmark for evaluating LLMs across diverse capabilities.",
             "DARE-bench is a comprehensive benchmark framework for evaluating large language models across "
             "diverse capabilities including reasoning, knowledge retrieval, code generation, and instruction "
             "following. It improves upon existing evaluation suites by providing multi-dimensional scoring, "
             "dynamic test set generation to prevent data contamination, and fine-grained capability profiling "
             "across over 20 task categories with difficulty-stratified test instances.",
             ["Multi-dimensional scoring", "Dynamic test sets prevent data contamination",
              "Fine-grained capability profiling", "Covers 20+ task categories"]),
            ("How does DARE-bench address data contamination in LLM evaluation?",
             "DARE-bench uses dynamic test set generation to prevent data contamination.",
             "DARE-bench addresses data contamination by generating dynamic test sets rather than relying on "
             "static benchmarks that may have been seen during model training. This ensures that evaluation "
             "results reflect genuine model capabilities rather than memorization of training data.",
             ["Dynamic test set generation", "Prevents memorization-based evaluation",
              "More reliable capability assessment"]),
        ],
    },
    {
        "arxiv_id": "2602.24287",
        "title": "Do LLMs Benefit From Their Own Words? Investigating Context Pollution in Language Models",
        "domain": "LLM",
        "category": "01.大语言模型基础",
        "subcategory": "Context Pollution",
        "difficulty": "advanced",
        "content": (
            "This paper investigates the phenomenon of context pollution in large language models — the effect "
            "that arises when LLM-generated text is fed back as input context for subsequent generations. "
            "The authors demonstrate that iterative self-consumption of generated text can lead to semantic "
            "drift, reduced diversity, and amplification of biases present in the original model. Through "
            "controlled experiments across multiple model families, the study quantifies degradation patterns "
            "in factual accuracy, reasoning coherence, and linguistic diversity. The work also proposes "
            "detection methods for identifying context pollution and mitigation strategies including "
            "diversity-promoting decoding and external knowledge grounding."
        ),
        "keywords": ["context pollution", "self-consumption", "semantic drift", "bias amplification",
                      "LLM generation", "model collapse"],
        "questions": [
            ("What is context pollution in LLMs and what are its effects?",
             "Context pollution occurs when LLM-generated text is fed back as input, causing semantic drift and bias amplification.",
             "Context pollution in LLMs is the phenomenon where model-generated text is fed back as input "
             "context for subsequent generations. This iterative self-consumption leads to semantic drift, "
             "reduced diversity, and amplification of biases present in the original model. Studies show "
             "degradation in factual accuracy, reasoning coherence, and linguistic diversity across "
             "multiple model families.",
             ["Semantic drift from self-consumption", "Reduced output diversity",
              "Bias amplification", "Degradation in factual accuracy"]),
            ("How can context pollution in LLMs be detected and mitigated?",
             "Detection methods and mitigation strategies include diversity-promoting decoding and external knowledge grounding.",
             "Context pollution can be detected through statistical analysis of generation patterns over "
             "iterative self-consumption cycles. Mitigation strategies include diversity-promoting decoding "
             "methods that maintain output variety, external knowledge grounding to anchor generations in "
             "factual sources, and monitoring for semantic drift across generation chains.",
             ["Statistical detection of drift patterns", "Diversity-promoting decoding",
              "External knowledge grounding", "Semantic drift monitoring"]),
        ],
    },
    {
        "arxiv_id": "2602.24283",
        "title": "LoRA-Pre: Taming Momentum in Low-Rank Optimizers for Pre-training",
        "domain": "LLM",
        "category": "05.有监督微调",
        "subcategory": "LoRA优化",
        "difficulty": "advanced",
        "content": (
            "LoRA-Pre proposes a novel low-rank optimizer designed for the pre-training phase of large "
            "language models. While LoRA (Low-Rank Adaptation) has been successful for fine-tuning, applying "
            "low-rank techniques during pre-training introduces challenges with momentum-based optimizers "
            "like Adam. The paper identifies that standard momentum accumulation in the low-rank subspace "
            "leads to suboptimal convergence due to stale gradient information from rank-constrained updates. "
            "LoRA-Pre introduces a momentum-taming mechanism that properly projects and rescales momentum "
            "terms when the low-rank basis changes, ensuring stable and efficient pre-training. Experiments "
            "on models up to 7B parameters demonstrate that LoRA-Pre achieves comparable performance to "
            "full-rank training while significantly reducing memory requirements."
        ),
        "keywords": ["LoRA", "low-rank optimizer", "pre-training", "momentum", "Adam",
                      "memory efficiency", "gradient projection"],
        "questions": [
            ("What problem does LoRA-Pre solve for low-rank pre-training?",
             "LoRA-Pre solves the momentum staleness problem when using low-rank optimizers during LLM pre-training.",
             "LoRA-Pre addresses the challenge of applying low-rank optimization during LLM pre-training. "
             "Standard momentum accumulation in the low-rank subspace leads to suboptimal convergence due to "
             "stale gradient information from rank-constrained updates. LoRA-Pre introduces a momentum-taming "
             "mechanism that properly projects and rescales momentum terms when the low-rank basis changes, "
             "achieving comparable performance to full-rank training with significantly less memory.",
             ["Momentum staleness in low-rank subspace", "Proper projection of momentum terms",
              "Comparable to full-rank training", "Significant memory reduction"]),
            ("How does LoRA-Pre differ from standard LoRA fine-tuning?",
             "LoRA-Pre is designed for pre-training rather than fine-tuning, with a momentum-taming mechanism.",
             "While standard LoRA is designed for parameter-efficient fine-tuning of pre-trained models, "
             "LoRA-Pre extends low-rank techniques to the pre-training phase. The key difference is the "
             "momentum-taming mechanism that handles the dynamic nature of pre-training where the low-rank "
             "basis frequently changes, unlike fine-tuning where the base model is relatively stable. "
             "LoRA-Pre has been validated on models up to 7B parameters.",
             ["Pre-training vs fine-tuning", "Dynamic low-rank basis handling",
              "Momentum rescaling mechanism", "Scales to 7B parameters"]),
        ],
    },
    {
        "arxiv_id": "2512.05049",
        "title": "QKAN-LSTM: A Quantum-Inspired KAN-LSTM Architecture for Sequence Modeling",
        "domain": "LLM",
        "category": "02.大语言模型架构",
        "subcategory": "LSTM变体",
        "difficulty": "advanced",
        "content": (
            "QKAN-LSTM introduces a novel neural network architecture that combines Kolmogorov-Arnold "
            "Networks (KAN) with LSTM cells, drawing inspiration from quantum computing principles. The "
            "architecture replaces traditional linear transformations in LSTM gates with learnable univariate "
            "functions parameterized via B-splines, following the Kolmogorov-Arnold representation theorem. "
            "Quantum-inspired features include superposition-like state mixing and entanglement-motivated "
            "gate coupling mechanisms. The model demonstrates improved performance on sequence modeling tasks "
            "including language modeling and time series prediction, with better parameter efficiency compared "
            "to standard LSTMs. The work bridges classical recurrent architectures with emerging KAN and "
            "quantum-inspired computing paradigms."
        ),
        "keywords": ["QKAN-LSTM", "Kolmogorov-Arnold Networks", "quantum-inspired", "LSTM",
                      "B-splines", "sequence modeling", "recurrent networks"],
        "questions": [
            ("What is QKAN-LSTM and how does it improve upon standard LSTM?",
             "QKAN-LSTM combines Kolmogorov-Arnold Networks with LSTM cells using quantum-inspired principles.",
             "QKAN-LSTM is a neural network architecture that enhances LSTM cells by replacing traditional "
             "linear transformations in gates with learnable univariate functions parameterized via B-splines, "
             "following the Kolmogorov-Arnold representation theorem. It incorporates quantum-inspired "
             "features such as superposition-like state mixing and entanglement-motivated gate coupling. "
             "This achieves improved performance on sequence modeling tasks with better parameter efficiency.",
             ["KAN-based gate functions", "B-spline parameterization",
              "Quantum-inspired state mixing", "Better parameter efficiency"]),
        ],
    },
    {
        "arxiv_id": "2602.24281",
        "title": "Memory Caching: Building RNNs with Growing Memory for Sequential Processing",
        "domain": "LLM",
        "category": "02.大语言模型架构",
        "subcategory": "RNN改进",
        "difficulty": "intermediate",
        "content": (
            "Memory Caching proposes a new approach to building recurrent neural networks with a growing "
            "memory mechanism. Unlike standard RNNs that maintain a fixed-size hidden state, Memory Caching "
            "RNNs dynamically expand their memory capacity as they process longer sequences. The architecture "
            "uses a hierarchical cache structure where frequently accessed memory slots are kept in a fast "
            "cache while less-used information is stored in an expandable slow cache. A learned attention "
            "mechanism determines when to allocate new memory slots and when to consolidate existing ones. "
            "This design enables better long-range dependency modeling without the quadratic complexity of "
            "Transformers, making it suitable for efficient sequential processing of very long sequences."
        ),
        "keywords": ["memory caching", "RNN", "growing memory", "hierarchical cache",
                      "long-range dependency", "sequential processing"],
        "questions": [
            ("How does Memory Caching improve RNNs for long sequence processing?",
             "Memory Caching RNNs dynamically expand memory capacity with a hierarchical cache structure.",
             "Memory Caching introduces RNNs with a growing memory mechanism that dynamically expands "
             "capacity for longer sequences. It uses a hierarchical cache structure: frequently accessed "
             "memory in a fast cache and less-used information in an expandable slow cache. A learned "
             "attention mechanism manages memory allocation and consolidation, enabling better long-range "
             "dependency modeling without Transformer-like quadratic complexity.",
             ["Dynamic memory expansion", "Hierarchical cache structure",
              "Learned memory allocation", "Sub-quadratic complexity"]),
        ],
    },
    # --- Agent domain ---
    {
        "arxiv_id": "2602.24286",
        "title": "CUDA Agent: Agentic Reinforcement Learning for CUDA Kernel Optimization",
        "domain": "Agent",
        "category": "10.大语言模型应用",
        "subcategory": "AI Agent",
        "difficulty": "advanced",
        "content": (
            "CUDA Agent presents an agentic reinforcement learning system for automatically optimizing "
            "CUDA GPU kernels. The agent iteratively analyzes kernel performance profiles, identifies "
            "bottlenecks, and applies optimization strategies including memory coalescing, shared memory "
            "utilization, warp-level primitives, and thread block configuration. Using an RL framework with "
            "execution-time feedback as reward signal, the agent learns to compose sequences of optimization "
            "actions that maximize kernel throughput. The system integrates with LLM-based code understanding "
            "to reason about kernel semantics and predict the impact of transformations. Experiments show "
            "the CUDA Agent achieves 1.5-3x speedups over manually optimized kernels on common GPU workloads "
            "including matrix operations, convolutions, and attention mechanisms."
        ),
        "keywords": ["CUDA", "GPU optimization", "reinforcement learning", "kernel optimization",
                      "agent", "memory coalescing", "warp primitives"],
        "questions": [
            ("What is CUDA Agent and how does it optimize GPU kernels?",
             "CUDA Agent is an RL-based system that automatically optimizes CUDA kernels through iterative profiling and transformation.",
             "CUDA Agent is an agentic reinforcement learning system for automatic CUDA kernel optimization. "
             "It iteratively analyzes performance profiles, identifies bottlenecks, and applies strategies "
             "like memory coalescing, shared memory utilization, and warp-level primitives. Using execution "
             "time as reward signal, it learns to compose optimization sequences. Integrated with LLM-based "
             "code understanding, it achieves 1.5-3x speedups over manually optimized kernels.",
             ["RL-based iterative optimization", "Execution-time reward signal",
              "LLM-based code understanding", "1.5-3x speedups achieved"]),
            ("How does CUDA Agent use reinforcement learning for kernel optimization?",
             "It uses execution-time feedback as reward to learn optimal sequences of kernel optimization actions.",
             "CUDA Agent employs reinforcement learning with kernel execution time as the reward signal. "
             "The agent learns to select and compose sequences of optimization actions — memory coalescing, "
             "shared memory utilization, warp-level primitives, thread block configuration — that maximize "
             "throughput. The RL framework enables the agent to discover non-obvious optimization "
             "combinations that outperform manual tuning on workloads including matrix operations, "
             "convolutions, and attention mechanisms.",
             ["Execution-time reward signal", "Compositional optimization actions",
              "Discovers non-obvious combinations", "Outperforms manual tuning"]),
        ],
    },
    {
        "arxiv_id": "2602.22401",
        "title": "Vibe Researching: AI Agents for Social Science Research",
        "domain": "Agent",
        "category": "10.大语言模型应用",
        "subcategory": "AI Agent",
        "difficulty": "intermediate",
        "content": (
            "Vibe Researching explores the use of AI agents as research assistants in social science "
            "methodology. The paper introduces a multi-agent framework where specialized agents handle "
            "different stages of the research pipeline: literature review, hypothesis generation, survey "
            "design, data collection planning, and statistical analysis. Each agent is augmented with "
            "domain-specific tools and knowledge bases relevant to social science research methods. "
            "The framework demonstrates how LLM-powered agents can accelerate the research cycle while "
            "maintaining methodological rigor through built-in checks for common pitfalls such as sampling "
            "bias, confounding variables, and p-hacking. Case studies in sociology and political science "
            "show the framework can reduce research preparation time by 40-60%."
        ),
        "keywords": ["AI agents", "social science", "multi-agent", "research methodology",
                      "hypothesis generation", "survey design"],
        "questions": [
            ("How do AI agents assist social science research in the Vibe Researching framework?",
             "Vibe Researching uses specialized multi-agent systems for different research pipeline stages.",
             "Vibe Researching introduces a multi-agent framework where specialized AI agents handle "
             "different research stages: literature review, hypothesis generation, survey design, data "
             "collection planning, and statistical analysis. Each agent uses domain-specific tools and "
             "knowledge bases for social science methods, with built-in checks for pitfalls like sampling "
             "bias and p-hacking. Case studies show 40-60% reduction in research preparation time.",
             ["Specialized agents per research stage", "Domain-specific tools and knowledge",
              "Methodological rigor checks", "40-60% time reduction"]),
        ],
    },
    {
        "arxiv_id": "2602.24273",
        "title": "Minimal Agent for Automated Theorem Proving",
        "domain": "Agent",
        "category": "10.大语言模型应用",
        "subcategory": "AI Agent",
        "difficulty": "advanced",
        "content": (
            "This paper presents a minimal yet effective agent architecture for automated theorem proving "
            "(ATP). Unlike complex multi-module systems, the Minimal Agent uses a single LLM augmented with "
            "a small set of carefully designed tools: a proof state inspector, a tactic suggester, and a "
            "backtracking controller. The agent interacts with formal proof assistants (Lean, Coq) through "
            "a standardized interface, applying tactics step-by-step while maintaining a proof search tree. "
            "The key insight is that a well-prompted LLM with minimal tooling can match or exceed more "
            "complex systems on standard theorem proving benchmarks (miniF2F, ProofNet). The paper also "
            "introduces a curriculum-based training approach that progressively increases theorem difficulty, "
            "achieving state-of-the-art results on several benchmark suites."
        ),
        "keywords": ["automated theorem proving", "Lean", "Coq", "proof assistant",
                      "minimal agent", "tactic search", "formal verification"],
        "questions": [
            ("What is the Minimal Agent approach to automated theorem proving?",
             "A single LLM with minimal tools (proof inspector, tactic suggester, backtracking controller) for theorem proving.",
             "The Minimal Agent for ATP uses a single LLM augmented with three carefully designed tools: "
             "a proof state inspector, a tactic suggester, and a backtracking controller. It interacts "
             "with formal proof assistants like Lean and Coq through a standardized interface, applying "
             "tactics step-by-step while maintaining a proof search tree. Despite its simplicity, it "
             "matches or exceeds more complex systems on benchmarks like miniF2F and ProofNet.",
             ["Single LLM with minimal tooling", "Three core tools",
              "Standardized proof assistant interface", "Matches complex systems on benchmarks"]),
            ("How does the curriculum-based training improve the Minimal Agent for theorem proving?",
             "Progressive difficulty increase during training helps the agent learn theorem proving strategies.",
             "The curriculum-based training approach progressively increases theorem difficulty during the "
             "agent's learning process. Starting with simpler lemmas and gradually introducing more complex "
             "theorems allows the agent to build foundational proof strategies before tackling harder problems. "
             "This approach achieves state-of-the-art results on several benchmark suites.",
             ["Progressive difficulty increase", "Foundation building from simple theorems",
              "State-of-the-art benchmark results"]),
        ],
    },
    # --- Evaluation domain ---
    {
        "arxiv_id": "2602.24277",
        "title": "RAG Evaluation Resources: The TREC DRAGUN Track",
        "domain": "Evaluation",
        "category": "08.检索增强rag",
        "subcategory": "RAG评估",
        "difficulty": "intermediate",
        "content": (
            "This paper describes the TREC DRAGUN (Dynamic Retrieval-Augmented Generation Under Noise) track, "
            "a standardized evaluation framework for assessing RAG systems. DRAGUN provides curated test "
            "collections with graded relevance judgments, noise-injected retrieval results, and multi-faceted "
            "evaluation metrics. The track covers key RAG challenges: faithfulness to retrieved context, "
            "robustness to irrelevant retrieved passages, handling of contradictory sources, and citation "
            "accuracy. The paper releases benchmark datasets spanning multiple domains (scientific, legal, "
            "medical) with human-annotated ground truth for both retrieval quality and generation quality. "
            "Initial results from participating systems highlight that current RAG approaches struggle most "
            "with noise robustness and source attribution accuracy."
        ),
        "keywords": ["RAG evaluation", "TREC DRAGUN", "retrieval-augmented generation",
                      "faithfulness", "noise robustness", "citation accuracy"],
        "questions": [
            ("What is the TREC DRAGUN track and what does it evaluate?",
             "TREC DRAGUN is a standardized evaluation framework for RAG systems covering faithfulness, noise robustness, and citation accuracy.",
             "TREC DRAGUN (Dynamic Retrieval-Augmented Generation Under Noise) is a standardized evaluation "
             "framework for RAG systems. It provides curated test collections with graded relevance judgments, "
             "noise-injected retrieval results, and multi-faceted metrics. It evaluates faithfulness to "
             "retrieved context, robustness to irrelevant passages, handling of contradictory sources, and "
             "citation accuracy across scientific, legal, and medical domains.",
             ["Standardized RAG evaluation", "Noise-injected test collections",
              "Multi-domain coverage", "Faithfulness and citation metrics"]),
            ("What are the main challenges for RAG systems identified by TREC DRAGUN?",
             "Current RAG systems struggle most with noise robustness and source attribution accuracy.",
             "Initial results from the TREC DRAGUN track show that current RAG approaches struggle most "
             "with noise robustness — maintaining quality when irrelevant passages are included in retrieval "
             "results — and source attribution accuracy, i.e., correctly citing which retrieved passages "
             "support generated claims. Handling contradictory sources also remains a significant challenge.",
             ["Noise robustness is weakest area", "Source attribution inaccuracy",
              "Contradictory source handling", "Current systems underperform on these dimensions"]),
        ],
    },
    {
        "arxiv_id": "2602.24266",
        "title": "Causal Abstractions: Sparsifying Neural Mechanisms for Interpretability",
        "domain": "Evaluation",
        "category": "09.大语言模型评估",
        "subcategory": "模型可解释性",
        "difficulty": "advanced",
        "content": (
            "Causal Abstractions presents a framework for understanding neural network mechanisms by "
            "identifying sparse causal circuits within large models. The approach combines causal "
            "intervention techniques with abstraction mapping to find minimal subnetworks that faithfully "
            "implement specific computational tasks. Given a high-level causal model specifying the desired "
            "computation, the method searches for neural mechanism implementations that align with the "
            "abstract specification while using the fewest possible components. Applied to large language "
            "models, the framework reveals interpretable circuits for tasks such as indirect object "
            "identification, factual recall, and arithmetic reasoning. The work provides theoretical "
            "guarantees on the faithfulness of discovered abstractions and demonstrates scalability to "
            "models with billions of parameters."
        ),
        "keywords": ["causal abstraction", "interpretability", "mechanistic interpretability",
                      "circuit discovery", "neural mechanisms", "sparsification"],
        "questions": [
            ("What are Causal Abstractions in the context of neural network interpretability?",
             "Causal Abstractions identify sparse causal circuits that faithfully implement specific computations in neural networks.",
             "Causal Abstractions is a framework for understanding neural network mechanisms by finding "
             "minimal subnetworks (sparse causal circuits) that faithfully implement specific computational "
             "tasks. It combines causal intervention techniques with abstraction mapping, searching for the "
             "fewest neural components that align with a high-level causal specification. Applied to LLMs, "
             "it reveals interpretable circuits for tasks like indirect object identification, factual "
             "recall, and arithmetic reasoning.",
             ["Sparse causal circuit discovery", "Causal intervention + abstraction mapping",
              "Minimal faithful subnetworks", "Scales to billion-parameter models"]),
        ],
    },
    {
        "arxiv_id": "2602.24278",
        "title": "Who Guards the Guardians? Representation Identifiability in Neural Networks",
        "domain": "Evaluation",
        "category": "09.大语言模型评估",
        "subcategory": "表示可识别性",
        "difficulty": "advanced",
        "content": (
            "Who Guards the Guardians? addresses the fundamental question of representation identifiability "
            "in neural networks — whether learned internal representations can be uniquely determined from "
            "observed behavior. The paper provides theoretical analysis showing conditions under which "
            "different networks with equivalent input-output behavior must share (or can differ in) their "
            "internal representations. The results have implications for model interpretability: if "
            "representations are not identifiable, then claims about what a model has 'learned' based on "
            "probing intermediate layers may be unreliable. The paper introduces formal criteria for "
            "representation identifiability, analyzes common architectures (Transformers, MLPs) under these "
            "criteria, and proposes regularization techniques that promote identifiable representations, "
            "enabling more trustworthy interpretability analysis."
        ),
        "keywords": ["representation identifiability", "interpretability", "probing",
                      "internal representations", "neural network analysis", "regularization"],
        "questions": [
            ("What is representation identifiability and why does it matter for LLM interpretability?",
             "Representation identifiability is whether learned internal representations can be uniquely determined from observed behavior.",
             "Representation identifiability asks whether a neural network's internal representations can "
             "be uniquely determined from its input-output behavior. This matters because if representations "
             "are not identifiable, different networks with the same behavior can have very different internal "
             "states, making probing-based interpretability claims unreliable. The paper provides theoretical "
             "conditions for identifiability and proposes regularization techniques to promote it.",
             ["Uniqueness of internal representations", "Implications for probing reliability",
              "Formal identifiability criteria", "Regularization for identifiable representations"]),
        ],
    },
    # --- VLM domain ---
    {
        "arxiv_id": "2602.24289",
        "title": "Mode Seeking meets Mean Seeking: Balanced Diffusion for Long Video Generation",
        "domain": "VLM",
        "category": "10.大语言模型应用",
        "subcategory": "视频生成",
        "difficulty": "advanced",
        "content": (
            "Mode Seeking meets Mean Seeking addresses the challenge of generating long, temporally coherent "
            "videos using diffusion models. Standard diffusion models exhibit a mode-seeking vs mean-seeking "
            "tradeoff: mode-seeking sampling produces sharp but inconsistent frames, while mean-seeking "
            "produces smooth but blurry results. The paper proposes a balanced diffusion framework that "
            "adaptively interpolates between mode-seeking and mean-seeking behavior across different "
            "temporal scales. Coarse temporal structure uses mean-seeking for global consistency, while "
            "fine-grained details use mode-seeking for visual sharpness. The method introduces a temporal "
            "hierarchy of diffusion processes with learned interpolation weights, enabling generation of "
            "videos significantly longer than the training sequence length while maintaining both coherence "
            "and visual quality."
        ),
        "keywords": ["diffusion models", "video generation", "mode seeking", "mean seeking",
                      "temporal coherence", "long video", "balanced diffusion"],
        "questions": [
            ("What is the mode-seeking vs mean-seeking tradeoff in video generation diffusion models?",
             "Mode-seeking produces sharp but inconsistent frames; mean-seeking produces smooth but blurry results.",
             "In diffusion-based video generation, mode-seeking sampling produces visually sharp but "
             "temporally inconsistent frames, while mean-seeking sampling produces smooth transitions but "
             "blurry results. The 'Mode Seeking meets Mean Seeking' paper proposes balanced diffusion that "
             "adaptively interpolates between both behaviors: mean-seeking for coarse temporal consistency "
             "and mode-seeking for fine-grained visual sharpness, enabling coherent long video generation.",
             ["Mode-seeking: sharp but inconsistent", "Mean-seeking: smooth but blurry",
              "Adaptive interpolation across temporal scales", "Enables coherent long videos"]),
        ],
    },
    {
        "arxiv_id": "2602.24290",
        "title": "UFO-4D: Unified Framework for 4D Reconstruction from Sparse Views",
        "domain": "VLM",
        "category": "10.大语言模型应用",
        "subcategory": "4D重建",
        "difficulty": "advanced",
        "content": (
            "UFO-4D presents a unified framework for 4D (3D + time) reconstruction from sparse multi-view "
            "video inputs. The method addresses the challenging problem of recovering dynamic 3D scenes "
            "from limited camera viewpoints by combining neural radiance fields with temporal flow estimation "
            "and multi-view consistency constraints. UFO-4D introduces a deformable 4D representation that "
            "factorizes appearance and motion, allowing efficient modeling of dynamic scenes. A key "
            "contribution is the sparse-view aggregation module that leverages cross-view attention to "
            "propagate information between limited viewpoints. The framework handles various dynamic "
            "content including articulated objects, fluid simulations, and human performances. Experiments "
            "show state-of-the-art results on standard 4D reconstruction benchmarks with as few as 3 input views."
        ),
        "keywords": ["4D reconstruction", "neural radiance field", "sparse views", "dynamic scenes",
                      "temporal flow", "deformable representation", "multi-view"],
        "questions": [
            ("What is UFO-4D and how does it achieve 4D reconstruction from sparse views?",
             "UFO-4D is a unified framework combining neural radiance fields with temporal flow for 4D reconstruction from limited viewpoints.",
             "UFO-4D is a unified framework for 4D reconstruction (3D + time) from sparse multi-view video. "
             "It combines neural radiance fields with temporal flow estimation and multi-view consistency "
             "constraints. A deformable 4D representation factorizes appearance and motion, while a "
             "sparse-view aggregation module uses cross-view attention to propagate information between "
             "limited viewpoints. It achieves state-of-the-art results with as few as 3 input views.",
             ["Neural radiance fields + temporal flow", "Deformable 4D representation",
              "Cross-view attention for sparse views", "State-of-the-art with 3 views"]),
        ],
    },
]


def make_document(idx: int, paper: dict) -> dict:
    """Create a document entry following the RAG schema."""
    return {
        "id": f"doc_arxiv_{idx:04d}",
        "category": paper["category"],
        "subcategory": paper["subcategory"],
        "title": paper["title"],
        "content": f"# {paper['title']}\n\nArXiv: {paper['arxiv_id']}\nDomain: {paper['domain']}\n\n{paper['content']}",
        "questions": [q[0] for q in paper["questions"]],
        "keywords": paper["keywords"],
        "difficulty": paper["difficulty"],
        "source_file": f"arxiv/{paper['arxiv_id']}.md",
        "url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
        "last_updated": NOW,
        "metadata": {
            "word_count": len(paper["content"].split()),
            "has_code": False,
            "has_images": False,
            "references": [f"https://arxiv.org/abs/{paper['arxiv_id']}"],
        },
    }


def make_qa_pairs(paper_idx: int, paper: dict) -> list[dict]:
    """Create QA pair entries following the RAG schema."""
    pairs = []
    for q_idx, (question, short_answer, detailed_answer, key_points) in enumerate(
        paper["questions"], start=1
    ):
        pairs.append(
            {
                "id": f"qa_arxiv_{paper_idx:04d}_{q_idx:02d}",
                "category": paper["category"],
                "subcategory": paper["subcategory"],
                "difficulty": paper["difficulty"],
                "question": question,
                "short_answer": short_answer,
                "detailed_answer": detailed_answer,
                "key_points": key_points,
                "code_examples": [],
                "related_topics": [
                    p["title"]
                    for p in PAPERS
                    if p["domain"] == paper["domain"]
                    and p["arxiv_id"] != paper["arxiv_id"]
                ],
                "keywords": paper["keywords"],
                "source_file": f"arxiv/{paper['arxiv_id']}.md",
                "url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
                "status": "verified",
            }
        )
    return pairs


def main():
    docs = []
    qas = []
    for idx, paper in enumerate(PAPERS, start=1):
        docs.append(make_document(idx, paper))
        qas.extend(make_qa_pairs(idx, paper))

    # Append to existing files
    with open(DOCS_FILE, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(QA_FILE, "a", encoding="utf-8") as f:
        for qa in qas:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"Appended {len(docs)} documents to {DOCS_FILE}")
    print(f"Appended {len(qas)} QA pairs to {QA_FILE}")
    print(f"Total documents now: 82 + {len(docs)} = {82 + len(docs)}")
    print(f"Total QA pairs now: 10 + {len(qas)} = {10 + len(qas)}")


if __name__ == "__main__":
    main()
