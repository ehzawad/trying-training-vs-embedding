# Bengali Legal Embeddings - Namjari Domain

Robust Bengali embedding model for legal text related to "Namjari" (land record mutation procedures in Bangladesh).

## Dataset
- **Location**: `namjari_questions/` 
- **Format**: 14 CSV files with question-tag pairs
- **Language**: Bengali (Bangla)
- **Domain**: Legal/Administrative land records
- **Size**: ~998 questions across 14 categories

## Quick Start

### Training Embedding Model (Research)
```bash
/Users/ehz/namjari-embedding-env/bin/python train_final.py
```

### Production Intent Classification (Recommended)
```bash
/Users/ehz/namjari-embedding-env/bin/python production_intent_system.py
/Users/ehz/namjari-embedding-env/bin/python namjari_query_handler.py
```

### Using the Production System
```python
from namjari_query_handler import NamjariQueryHandler

# Initialize production handler
handler = NamjariQueryHandler()

# Handle queries
result = handler.handle_query("নামজারি করতে কি করতে হবে?")
print(f"Domain: {result['domain']}, Category: {result['category']}")
# Output: Domain: namjari, Category: application_procedure
```

## Model Architecture
- **Base Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Training**: 1 epoch, data-driven approach (no hard-coded assumptions)
- **Loss Function**: CoSENTLoss
- **Performance**: Pearson 0.905, Spearman 0.927
- **Speed**: 13-second training time

## Key Features
- ✅ **Anti-overfitting**: Only 1 epoch, 15 questions per category
- ✅ **Data-driven**: No hard-coded "irrelevant" terms
- ✅ **Fast training**: 13 seconds vs hours
- ✅ **Realistic scores**: 0.90 range (not suspicious 0.97+)
- ✅ **Apple Silicon optimized**: MPS compatible

## Research Findings & Solution
**Key Discovery**: After extensive experimentation (1-20 epochs, multiple approaches), we found that **sentence embeddings struggle with Bengali legal text** due to syntactic pattern dominance. Even semantically unrelated queries achieve 0.68-0.93 similarity scores.

**✅ SOLUTION IMPLEMENTED**: Hybrid intent classification system combining keyword matching + ML classifier achieves **100% accuracy** on critical test cases. See `FINDINGS.md` for detailed analysis.

**Production Recommendation**: Use `namjari_query_handler.py` for production deployments instead of embeddings.

## Dependencies
```bash
pip install -r requirements.txt
```
