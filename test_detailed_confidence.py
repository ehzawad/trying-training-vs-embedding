#!/usr/bin/env python3
"""
Test detailed confidence scores on user's specific examples
"""

import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from simple_pure_overfitting import SimplePureOverfittingClassifier
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detailed_confidence():
    """Test confidence scores on user's specific examples"""
    
    user_examples = [
        "জমি দখলের কিভাবে পেতে পারি?",
        "আমার হারিয়ে যাওয়া বই পেতে কি করতে হবে?",
        "হজ্ব  করতে চাই, কি করতে হবে?",
        "অনলাইনে ওমরাহ্‌ নিবন্ধন করতে কি করতে হবে?",
        "মিউট করতে হলে কি করতে হবে?",
        "আমি ওমরাহ্‌ করবো কি করতে হবে?",
        "কোম্পানিতে আবেদন কি নিজে করতে পারি।",
        "জমি দখল করতে কি ভূমি অফিসে যাওয়া লাগে?",
        "আমি নিজে কি চাকরির আবেদন করতে পারবো?",
        "কারো সাহায্যে কি দখল করা যাবে?",
        "জমি দখলের জন্য নিবন্ধন করতে কি লাগে?",
        "জন্মনিবন্ধন করতে কি কি দরকার?",
        "জন্মনিবন্ধন করতে কি দলিল লাগে?",
        "জন্মনিবন্ধন করতে নিজের মোবাইল থাকতে হবে?",
        "জন্মনিবন্ধন করতে মোবাইল নম্বর ছাড়া আর কি লাগে?",
        "জন্মনিবন্ধন করতে এনআইডি বা জন্মনিবন্ধন লাগে কি?",
        "হজ্ব আবেদন নিজে না করলে প্রতিনিধি দিয়ে করা যায় কিনা?",
        "আমি নিজ এলাকার বাইরে থাকি অন্য কেউ কি আমার নামে চাঁদাবাজি করতে পারে কিনা?",
        "আমি বিদেশে থাকি আমার ভাই বা আত্মীয় দশ পারসেন্ট আবেদন করতে পারে কিনা?",
        "আমি নারী জমির কিছুই বুঝি না, আমি সিনেসিস এ আবেদন করাতে পারবো তো?",
    ]
    
    # Load classifier
    logger.info("🔬 DETAILED CONFIDENCE TESTING")
    logger.info("="*80)
    
    classifier = SimplePureOverfittingClassifier()
    classifier.load_all_training_data()
    classifier.create_memory_index()
    
    # Get embeddings for all examples
    model = classifier.model
    example_embeddings = model.encode(user_examples)
    
    print("\n🎯 DETAILED CONFIDENCE ANALYSIS")
    print("="*80)
    
    high_confidence_namjari = []
    medium_confidence_namjari = []
    low_confidence_namjari = []
    
    for i, example in enumerate(user_examples):
        # Get top 5 most similar training examples
        query_embedding = example_embeddings[i:i+1]
        similarities, indices = classifier.faiss_index.search(query_embedding, k=5)
        
        max_similarity = similarities[0][0]
        
        # Get classification
        result = classifier.classify_with_extreme_thresholds(example)
        
        print(f"\n📝 Query: {example}")
        print(f"   🎯 Classification: {result['domain']} (confidence: {result['confidence']:.3f})")
        print(f"   📊 Max Similarity: {max_similarity:.4f}")
        print(f"   🔍 Top 5 Similar Training Examples:")
        
        for j in range(5):
            similar_idx = indices[0][j]
            similarity = similarities[0][j]
            similar_text = classifier.training_texts[similar_idx]
            similar_category = classifier.training_labels[similar_idx]
            print(f"      {j+1}. [{similarity:.4f}] ({similar_category}) {similar_text}")
        
        # Categorize by confidence
        if result['domain'] == 'namjari':
            if result['confidence'] > 0.9:
                high_confidence_namjari.append((example, result['confidence'], max_similarity))
            elif result['confidence'] > 0.7:
                medium_confidence_namjari.append((example, result['confidence'], max_similarity))
            else:
                low_confidence_namjari.append((example, result['confidence'], max_similarity))
    
    # Pattern Analysis
    print("\n\n🧠 PATTERN ANALYSIS")
    print("="*80)
    
    print(f"\n🔥 HIGH Confidence Namjari ({len(high_confidence_namjari)} examples):")
    for example, conf, sim in high_confidence_namjari:
        print(f"   • [{conf:.3f}, sim:{sim:.4f}] {example}")
    
    print(f"\n🔶 MEDIUM Confidence Namjari ({len(medium_confidence_namjari)} examples):")
    for example, conf, sim in medium_confidence_namjari:
        print(f"   • [{conf:.3f}, sim:{sim:.4f}] {example}")
    
    print(f"\n🔵 LOW Confidence Namjari ({len(low_confidence_namjari)} examples):")
    for example, conf, sim in low_confidence_namjari:
        print(f"   • [{conf:.3f}, sim:{sim:.4f}] {example}")
    
    print(f"\n📊 SIMILARITY DISTRIBUTION:")
    all_sims = [max_similarity for _, _, max_similarity in high_confidence_namjari + medium_confidence_namjari + low_confidence_namjari]
    if all_sims:
        print(f"   📈 Max similarity: {max(all_sims):.4f}")
        print(f"   📉 Min similarity: {min(all_sims):.4f}")
        print(f"   📊 Avg similarity: {sum(all_sims)/len(all_sims):.4f}")
    
    # Identify patterns
    print(f"\n🎯 PATTERN INSIGHTS:")
    
    # Keywords that appear in queries
    keywords_found = {}
    for example in user_examples:
        words = example.split()
        for word in words:
            if len(word) > 2:  # Skip short words
                keywords_found[word] = keywords_found.get(word, 0) + 1
    
    common_words = sorted(keywords_found.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   📝 Most common words:")
    for word, count in common_words:
        print(f"      • {word}: {count} times")
    
    # Suggest hyperparameter changes
    print(f"\n🚀 HYPERPARAMETER SUGGESTIONS FOR EXTREME OVERFITTING:")
    print(f"   1. Lower extreme_threshold to 0.85-0.90 (currently 0.93)")
    print(f"   2. Lower high_threshold to 0.80-0.85 (currently 0.90)")
    print(f"   3. Lower medium_threshold to 0.75-0.80 (currently 0.89)")
    print(f"   4. Use top-3 instead of top-1 similarity")
    print(f"   5. Add weighted averaging of top similarities")
    print(f"   6. Consider context-aware similarity (sentence structure)")
    
    return high_confidence_namjari, medium_confidence_namjari, low_confidence_namjari

if __name__ == "__main__":
    test_detailed_confidence()