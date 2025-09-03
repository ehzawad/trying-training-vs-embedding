#!/usr/bin/env python3
"""
Final Test Script for Bengali Legal Embeddings
Tests the trained model against out-of-scope queries and training data
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_final_model():
    """
    Comprehensive test of the final trained model
    """
    
    model_path = "models/namjari-final/final"
    
    try:
        logger.info(f"Loading final model: {model_path}")
        trained_model = SentenceTransformer(model_path)
    except Exception as e:
        logger.error(f"Model not found: {e}")
        logger.info("Please run train_final.py first")
        return None
    
    # Load base model for comparison
    base_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Test 1: In-scope Namjari queries
    namjari_queries = [
        "নামজারি করতে কি করতে হবে?",
        "নামজারির ফি কত টাকা?", 
        "নামজারি করতে কি দলিল লাগে?",
        "নামজারি আবেদন কোথায় করতে হয়?",
        "নামজারির স্ট্যাটাস চেক করতে চাই।"
    ]
    
    # Test 2: Out-of-scope queries (your critical test data)
    out_of_scope_queries = [
        "জমি দখলের কিভাবে পেতে পারি?",
        "আমার হারিয়ে যাওয়া বই পেতে কি করতে হবে?",
        "হজ্ব করতে চাই, কি করতে হবে?",
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
    
    def analyze_model_performance(model, model_name):
        logger.info(f"\n--- {model_name} Performance ---")
        
        # Get embeddings
        namjari_embeddings = model.encode(namjari_queries)
        out_of_scope_embeddings = model.encode(out_of_scope_queries)
        
        # Calculate similarities
        within_namjari_sim = cosine_similarity(namjari_embeddings, namjari_embeddings)
        cross_domain_sim = cosine_similarity(namjari_embeddings, out_of_scope_embeddings)
        
        # Remove diagonal for within-domain
        within_scores = within_namjari_sim[np.triu_indices_from(within_namjari_sim, k=1)]
        
        within_avg = within_scores.mean()
        cross_avg = cross_domain_sim.mean()
        separation = within_avg - cross_avg
        
        # Count problematic cases
        problematic_cases = (cross_domain_sim > 0.6).sum()
        total_cross_pairs = cross_domain_sim.size
        
        logger.info(f"  Within-domain similarity: {within_avg:.3f}")
        logger.info(f"  Cross-domain similarity: {cross_avg:.3f}")
        logger.info(f"  Domain separation: {separation:.3f}")
        logger.info(f"  Problematic cases: {problematic_cases}/{total_cross_pairs} ({problematic_cases/total_cross_pairs*100:.1f}%)")
        
        # Show worst offenders
        worst_cases = []
        for i, namjari_q in enumerate(namjari_queries):
            for j, oos_q in enumerate(out_of_scope_queries):
                sim = cross_domain_sim[i][j]
                if sim > 0.7:  # Very high similarity
                    worst_cases.append((namjari_q, oos_q, sim))
        
        if worst_cases:
            logger.info(f"  🚨 Worst cases (similarity > 0.7):")
            for namjari_q, oos_q, sim in sorted(worst_cases, key=lambda x: x[2], reverse=True)[:5]:
                logger.info(f"    {sim:.3f}: '{namjari_q[:30]}...' vs '{oos_q[:30]}...'")
        
        return {
            'within_domain': within_avg,
            'cross_domain': cross_avg,
            'separation': separation,
            'problematic_cases': problematic_cases,
            'total_pairs': total_cross_pairs
        }
    
    # Test both models
    logger.info("Testing model performance...")
    base_results = analyze_model_performance(base_model, "BASE MODEL")
    trained_results = analyze_model_performance(trained_model, "TRAINED MODEL")
    
    # Quick sanity check with realistic examples
    logger.info(f"\n--- Quick Sanity Check ---")
    
    test_cases = [
        # Should be similar (same concept)
        ("নামজারি করতে কি করতে হবে?", "ভূমি রেকর্ড পরিবর্তন করতে কি লাগে?"),
        
        # Should be moderately similar (related legal concepts)  
        ("নামজারির ফি কত?", "নামজারি করতে কি দলিল লাগে?"),
        
        # Should be low similarity (unrelated)
        ("নামজারি করতে কি করতে হবে?", "আজকের আবহাওয়া কেমন?"),
    ]
    
    for i, (q1, q2) in enumerate(test_cases):
        emb1 = trained_model.encode([q1])
        emb2 = trained_model.encode([q2])
        sim = cosine_similarity(emb1, emb2)[0][0]
        
        logger.info(f"  Test {i+1}: {sim:.3f}")
        logger.info(f"    Q1: {q1}")
        logger.info(f"    Q2: {q2}")
    
    return trained_results, base_results

if __name__ == "__main__":
    try:
        trained_results, base_results = test_final_model()
        
        if trained_results and base_results:
            print("\n" + "="*80)
            print("🎯 FINAL MODEL TEST RESULTS")
            print("="*80)
            print("📊 Trained Model Performance:")
            print(f"   Within-domain similarity: {trained_results['within_domain']:.3f}")
            print(f"   Cross-domain similarity: {trained_results['cross_domain']:.3f}")
            print(f"   Domain separation: {trained_results['separation']:.3f}")
            print(f"   Problematic cases: {trained_results['problematic_cases']}/{trained_results['total_pairs']}")
            
            improvement = ((base_results['cross_domain'] - trained_results['cross_domain']) / base_results['cross_domain']) * 100
            print(f"   Improvement vs base: {improvement:.1f}%")
            
            print("\n🔍 Key Findings:")
            if trained_results['separation'] > 0.2:
                print("   ✅ Reasonable domain separation achieved")
            else:
                print("   ⚠️ Weak domain separation (Bengali syntactic challenge)")
                
            if trained_results['problematic_cases'] < 20:
                print("   ✅ Few problematic out-of-scope cases")
            else:
                print("   ⚠️ Many problematic out-of-scope cases (expected for Bengali)")
            
            print(f"\n📝 See FINDINGS.md for detailed analysis and alternative approaches")
            print("="*80)
            
    except Exception as e:
        logger.error(f"Final test failed: {e}")
        raise
