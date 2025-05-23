# 🔍 LightRAG: Enhanced Graph-Based Retrieval-Augmented Generation System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/Research_Paper-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/INFO6105-FF5722?style=for-the-badge"/>
</div>

## 📋 Project Overview

**LightRAG** is an enhanced implementation of the Light Retrieval-Augmented Generation system, developed as part of the INFO6105 Data Science Engineering Methods and Tools course at Northeastern University. This project addresses critical limitations in traditional RAG systems through graph-based knowledge representation and innovative document processing techniques.

### 🎯 Key Achievements
- **96.9% reduction** in document fragmentation through adaptive content-aware chunking
- **83% reduction** in system latency compared to GraphRAG baseline
- **Cross-parameter consistency** with stable performance across different chunk sizes (500, 1000, 2000 characters)
- **Enhanced retrieval quality** through structure-preserving document processing
- **Comprehensive evaluation** demonstrating superiority over existing RAG approaches

## 🚀 Core Innovation

### **Graph-Based Knowledge Representation**
Unlike traditional RAG systems that treat documents as isolated chunks, LightRAG constructs interconnected knowledge graphs that capture complex relationships between entities, enabling more contextually accurate and comprehensive responses.

### **Adaptive Content-Aware Chunking Strategy**
Our novel chunking approach preserves document structure and semantic coherence:
- **Structure-Preserving Processing**: Respects paragraph and sentence boundaries
- **Semantic Unit Preservation**: Maintains complete semantic units to preserve meaning
- **Cross-Parameter Consistency**: Stable performance regardless of chunk size configuration

## 🏗️ Repository Structure

```
📂 LightRAG-Enhanced/
├── 🔧 lightrag/
│   ├── __init__.py           # Package initialization
│   ├── base.py               # Base classes and core functionality
│   ├── lightrag.py           # Main LightRAG implementation
│   ├── improved.py           # Enhanced features and optimizations
│   └── utils.py              # Utility functions and helpers
├── 🧪 tests/
│   ├── test_chunking.py      # Chunking strategy tests
│   ├── test_retrieval.py     # Retrieval mechanism tests
│   └── test_integration.py   # End-to-end integration tests
├── 📊 evaluation/
│   ├── chunking_eval.py      # Chunking strategy comparison
│   ├── performance_test.py   # Performance benchmarking
│   └── results/              # Evaluation results and visualizations
├── 📁 examples/
│   ├── basic_usage.py        # Simple implementation example
│   └── advanced_config.py    # Advanced configuration examples
├── 📋 requirements.txt       # Project dependencies
├── 📖 README.md              # Project documentation
└── 📄 LightRAG Graph-Based Retrieval-Augmented Generation System.pdf
```

## 📊 Performance Results

### **Enhanced Chunking Performance**
Our adaptive content-aware chunking strategy demonstrates significant improvements:

| Chunk Size | Baseline Broken Sentences | Enhanced Broken Sentences | Improvement |
|------------|---------------------------|---------------------------|-------------|
| **500 characters** | 277 | 5 | **98.2%** |
| **1000 characters** | 138 | 5 | **96.4%** |
| **2000 characters** | 69 | 5 | **92.8%** |
| **Average** | **161.3** | **5** | **96.9%** |

### **System Architecture Benefits**
- **Dual-Level Retrieval**: Combined low-level entity and high-level conceptual retrieval
- **Graph-Based Indexing**: Captures complex entity relationships for better context
- **Incremental Updates**: Efficient integration of new information without rebuilding
- **Semantic Coherence**: Better preservation of document meaning and structure

## 💻 Technical Implementation

### **Enhanced Chunking Algorithm**
```python
def improved_chunking(text, chunk_size=1000):
    """
    Adaptive content-aware chunking that preserves semantic integrity
    """
    # Split text into paragraphs
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Check if adding this paragraph exceeds the chunk size
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            # Add separator if needed
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
            current_chunk += para
    
    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
```

### **Core Technologies**
- **Language Model**: Google Gemini 1.5 Flash for entity extraction and response generation
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2) for vector representations
- **Graph Processing**: Custom graph implementation with efficient in-memory storage
- **Vector Database**: Integrated similarity search with graph linkage

## 🔧 Installation & Quick Start

### **Environment Setup**
```bash
# Clone the repository
git clone https://github.com/ER-Bhagyashri-Pagar/LightRAG-Enhanced-Graph-Based-Retrieval-Augmented-Generation-System.git
cd LightRAG-Enhanced-Graph-Based-Retrieval-Augmented-Generation-System

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Basic Usage**
```python
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import asyncio

# Initialize LightRAG with enhanced features
async def initialize():
    rag = LightRAG(
        working_dir="./my_lightrag",
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    return rag

# Use enhanced chunking
from lightrag.improved import adaptive_chunking
chunks = adaptive_chunking(text, chunk_size=1000)
```

## 📚 Documentation & Resources

### **Comprehensive Technical Report**
📄 **[LightRAG Graph-Based RAG System Documentation](https://github.com/ER-Bhagyashri-Pagar/LightRAG-Enhanced-Graph-Based-Retrieval-Augmented-Generation-System/blob/main/LightRAG%20Graph-Based%20Retrieval-Augmented%20Generation%20System.pdf)**

*Complete technical documentation covering:*
- Theoretical background and research methodology
- Detailed implementation architecture
- Comprehensive experimental evaluation and results
- Performance analysis and comparative benchmarking
- Novel enhancements and algorithmic contributions

### **Research Foundation**
This implementation builds upon:
- **Original Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" by Guo et al. (2024)
- **Advanced RAG Systems**: Comprehensive literature review and comparative analysis
- **Graph-Based Knowledge Representation**: State-of-the-art approaches in knowledge graphs

## 🎯 Key Research Contributions

### **1. Enhanced Document Processing**
- **96.9% improvement** in maintaining semantic coherence through adaptive chunking
- **Structure-preserving algorithm** that respects document boundaries
- **Cross-parameter consistency** ensuring stable performance across configurations

### **2. Implementation Excellence**
- Successfully implemented complete LightRAG architecture with significant optimizations
- **Comprehensive evaluation** against baseline methods (NaiveRAG, RQ-RAG, HyDE, GraphRAG)
- **Quantifiable improvements** in retrieval quality and computational efficiency

### **3. Academic Rigor**
- **Rigorous experimental methodology** with proper baseline comparisons
- **Statistical significance** demonstrated through comprehensive testing
- **Reproducible research** with detailed implementation documentation

## 🔮 Future Research Directions

### **Planned Enhancements**
1. **Semantic-Guided Chunking**: Further refinement based on semantic similarity
2. **Multi-Document Knowledge Linking**: Enhanced cross-document relationship detection
3. **Dynamic Retrieval Optimization**: Adaptive parameter tuning based on query characteristics
4. **Comprehensive Evaluation Framework**: Standardized benchmarks for RAG performance

### **Advanced Applications**
- **Multi-Modal Integration**: Extending to images, audio, and structured data
- **Domain-Specific Adaptation**: Specialized configurations for different knowledge domains
- **Real-Time Learning**: Continuous adaptation to new information streams

## 🏆 Project Impact

### **Academic Significance**
- **Graduate-level research** demonstrating advanced data science capabilities
- **Novel algorithmic contributions** to the field of retrieval-augmented generation
- **Comprehensive documentation** supporting reproducible research

### **Technical Excellence**
- **Quantifiable performance improvements** with rigorous evaluation methodology
- **Production-ready implementation** suitable for real-world applications
- **Extensible architecture** supporting future research and development

---

<div align="center">
  <p><strong>🔬 Advanced Research in Graph-Based RAG Systems</strong></p>
  <p><em>Northeastern University | INFO6105 Data Science Engineering Methods and Tools</em></p>
  <p><em>Bridging the gap between knowledge representation and intelligent information retrieval</em></p>
</div>
