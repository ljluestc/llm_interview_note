# LLM Interview Notes - RAG System

Complete Retrieval-Augmented Generation (RAG) system for semantic search over LLM interview documentation.

## 📁 Directory Structure

```
data/
├── RAG_SCHEMA.md           # Dataset schema documentation
├── README.md               # This file
├── raw/                    # Raw data (if needed)
├── processed/              # Processed JSONL datasets
│   ├── all_documents.jsonl       # All documentation (82 documents)
│   ├── all_qa_pairs.jsonl        # Q&A pairs (10 pairs)
│   └── dataset_summary.json      # Dataset statistics
└── embeddings/             # Vector embeddings
    ├── doc_embeddings.npy        # Document embeddings
    └── qa_embeddings.npy         # Q&A embeddings

scripts/
└── convert_md_to_rag.py    # Markdown to JSONL converter

rag_system/
└── rag_engine.py           # Complete RAG engine implementation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Convert Markdown to RAG Format (Already Done)

```bash
cd scripts
python convert_md_to_rag.py --input-dir ../ --output-dir ../data/processed
```

### 3. Use RAG Engine

```python
from rag_system.rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine()

# Load processed data
rag.load_data('data/processed')

# Generate embeddings (first time only)
rag.generate_embeddings(save_to='data/embeddings')

# Build search index
rag.build_index()

# Search for relevant content
results = rag.search("什么是attention机制?", top_k=5)

# Print results
for i, result in enumerate(results):
    print(f"[{i+1}] Score: {result['score']:.4f}")
    print(f"Title: {result.get('title') or result.get('question')}")
    print()
```

### 4. Generate Answers with Context

```python
# Get answer with sources
answer_data = rag.generate_answer("什么是LoRA?", top_k=3)

print(f"Query: {answer_data['query']}")
print(f"\nContext:\n{answer_data['context']}")
print(f"\nSources:")
for source in answer_data['sources']:
    print(f"  - {source['title']} (score: {source['score']:.4f})")
```

## 📊 Dataset Statistics

- **Total Documents**: 82
- **Total Q&A Pairs**: 10
- **Categories**: 11
  - 01.大语言模型基础 (LLM Basics)
  - 02.大语言模型架构 (LLM Architecture)
  - 03.训练数据集 (Training Datasets)
  - 04.分布式训练 (Distributed Training)
  - 05.有监督微调 (Supervised Fine-tuning)
  - 06.推理 (Inference)
  - 07.强化学习 (Reinforcement Learning)
  - 08.检索增强RAG (RAG)
  - 09.大语言模型评估 (LLM Evaluation)
  - 10.大语言模型应用 (LLM Applications)

## 🔧 Advanced Usage

### Custom Embedding Model

```python
# Use different embedding model
rag = RAGEngine(
    model_name='moka-ai/m3e-base',  # Chinese-optimized model
    device='cuda'  # Use GPU if available
)
```

### Search Options

```python
# Search only in Q&A pairs
results = rag.search("如何微调模型?", top_k=5, search_type='qa')

# Search only in documents
results = rag.search("Transformer架构", top_k=5, search_type='documents')

# Filter by minimum score
results = rag.search("什么是PPO?", top_k=5, min_score=0.5)
```

### Reranking for Better Results

```python
# Initial search
results = rag.search("什么是attention?", top_k=10)

# Rerank for improved relevance
reranked = rag.rerank("什么是attention?", results, top_k=5)
```

### Load Pre-computed Embeddings

```python
# Skip embedding generation if already computed
rag = RAGEngine()
rag.load_data('data/processed')
rag.load_embeddings('data/embeddings')  # Load from disk
rag.build_index()
```

## 🛠️ Scripts

### convert_md_to_rag.py

Converts markdown documentation to RAG-ready JSONL format.

**Features:**
- Extracts title, sections, and content from markdown
- Identifies Q&A content automatically
- Extracts code blocks and keywords
- Infers difficulty levels
- Generates unique IDs with category prefixes
- Creates comprehensive metadata

**Usage:**
```bash
python convert_md_to_rag.py \
    --input-dir ../ \
    --output-dir ../data/processed \
    --base-url http://wdndev.github.io/llm_interview_note
```

## 📋 Schema

See [RAG_SCHEMA.md](RAG_SCHEMA.md) for detailed schema documentation.

### Document Schema

```json
{
  "id": "doc_01_0001",
  "category": "01.大语言模型基础",
  "subcategory": "attention",
  "title": "Attention机制详解",
  "content": "...",
  "questions": ["什么是attention?", "..."],
  "keywords": ["attention", "transformer", "..."],
  "difficulty": "intermediate",
  "source_file": "01.大语言模型基础/1.attention/1.attention.md",
  "url": "http://...",
  "last_updated": "2024-03-07T10:00:00",
  "metadata": {
    "word_count": 5000,
    "has_code": true,
    "has_images": true,
    "references": []
  }
}
```

### Q&A Schema

```json
{
  "id": "qa_02_0001",
  "category": "02.大语言模型架构",
  "subcategory": "attention",
  "difficulty": "intermediate",
  "question": "什么是self-attention?",
  "short_answer": "Self-attention是一种...",
  "detailed_answer": "详细解释...",
  "key_points": ["点1", "点2", "..."],
  "code_examples": ["code snippet"],
  "related_topics": ["multi-head attention", "..."],
  "keywords": ["self-attention", "query", "key", "value"],
  "source_file": "...",
  "url": "...",
  "status": "verified"
}
```

## 🔍 Example Queries

```python
# 1. Basic concept questions
rag.search("什么是Transformer?")
rag.search("解释attention机制")

# 2. Technical implementation
rag.search("如何实现LoRA微调?")
rag.search("vLLM推理优化技术")

# 3. Comparison questions
rag.search("LoRA和Adapter-tuning的区别")
rag.search("数据并行vs流水线并行")

# 4. Troubleshooting
rag.search("如何解决显存不足问题?")
rag.search("模型幻觉如何缓解?")
```

## 🌟 Features

1. **Multilingual Support**: Optimized for Chinese and English content
2. **Fast Search**: FAISS-accelerated vector search (falls back to numpy)
3. **Hybrid Search**: Combines semantic and keyword-based search
4. **Reranking**: Improves result relevance
5. **Source Attribution**: Tracks sources for each result
6. **Flexible Schema**: Easy to extend with new fields
7. **Batch Processing**: Efficient embedding generation
8. **Persistent Storage**: Save/load embeddings to avoid recomputation

## 📈 Performance

- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2`
  - Dimensions: 768
  - Languages: 50+
  - Speed: ~100 docs/sec (CPU)

- **Search Speed**:
  - With FAISS: <10ms for 100k documents
  - Without FAISS: ~50ms for 1k documents

## 🔮 Future Enhancements

- [ ] Add cross-encoder reranking
- [ ] Implement hybrid search (BM25 + semantic)
- [ ] Add query expansion
- [ ] Support for image/diagram retrieval
- [ ] Integration with LLM for answer generation
- [ ] Add caching layer
- [ ] Implement incremental updates
- [ ] Add evaluation metrics (MRR, NDCG)

## 📚 References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## 🤝 Contributing

To add new content:

1. Add markdown files to appropriate categories
2. Run converter: `python scripts/convert_md_to_rag.py`
3. Regenerate embeddings: `rag.generate_embeddings(save_to='data/embeddings')`
4. Rebuild index: `rag.build_index()`

## 📄 License

Same as the main repository.

---

**Note**: The embeddings are not included in git due to size. Run `rag.generate_embeddings()` on first use.
