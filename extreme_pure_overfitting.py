#!/usr/bin/env python3
"""
EXTREME Pure Overfitting - Hyperparameter Tuned for Maximum Overfitting
Based on pattern analysis, this uses aggressive thresholds to reject more out-of-scope
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import torch
from sentence_transformers import SentenceTransformer
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtremePureOverfittingClassifier:
    """
    EXTREME pure overfitting with hyperparameter tuning
    Strategy: Even more aggressive similarity thresholds + top-K averaging
    """
    
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Training data storage
        self.training_texts = []
        self.training_labels = []
        self.training_embeddings = None
        self.faiss_index = None
        
        # EXTREME thresholds based on pattern analysis
        # Analysis showed similarities 2.0-6.0, so we need MUCH higher thresholds
        self.ultra_extreme_threshold = 0.98  # Nearly identical (99.8%+ similarity)
        self.extreme_threshold = 0.95        # Almost identical (95%+ similarity) 
        self.very_high_threshold = 0.90      # Very high similarity (90%+)
        self.high_threshold = 0.85           # High similarity (85%+)
        
        # Advanced parameters for extreme overfitting
        self.use_top_k_averaging = True      # Average top-3 similarities
        self.top_k = 3                       # Use top-3 matches
        self.require_category_consensus = True  # Require majority category agreement
        self.similarity_boost_factor = 1.1    # Boost similarity scores slightly
        
        logger.info("🔥 Initialized EXTREME Pure Overfitting Classifier")
        logger.info(f"   Ultra-Extreme: {self.ultra_extreme_threshold}")
        logger.info(f"   Extreme: {self.extreme_threshold}")
        logger.info(f"   Very-High: {self.very_high_threshold}")  
        logger.info(f"   High: {self.high_threshold}")
    
    def load_all_training_data(self, data_dir: str = "namjari_questions"):
        """Load ALL training data for memorization"""
        logger.info("Loading ALL training data for EXTREME memorization...")
        
        data_dir = Path(data_dir)
        all_texts = []
        all_labels = []
        
        csv_files = list(data_dir.glob("*.csv"))
        for csv_file in csv_files:
            category = csv_file.stem
            df = pd.read_csv(csv_file)
            
            if 'question' in df.columns:
                questions = df['question'].tolist()
                labels = [category] * len(questions)
                
                all_texts.extend(questions)
                all_labels.extend(labels)
                
                logger.info(f"  {category}: {len(questions)} examples")
        
        self.training_texts = all_texts
        self.training_labels = all_labels
        
        logger.info(f"Total: {len(self.training_texts)} training examples loaded")
        return self.training_texts, self.training_labels
    
    def create_memory_index(self):
        """Create FAISS memory index for EXTREME overfitting"""
        logger.info("Creating EXTREME memory index...")
        
        # Encode all training data
        self.training_embeddings = self.model.encode(
            self.training_texts, 
            convert_to_tensor=False, 
            show_progress_bar=True
        )
        
        # Create FAISS index for cosine similarity
        dimension = self.training_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        embeddings_normalized = self.training_embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        self.faiss_index.add(embeddings_normalized)
        
        logger.info(f"EXTREME memory index created with {self.faiss_index.ntotal} training examples")
        return self.faiss_index
    
    def classify_with_extreme_overfitting(self, query: str, k: int = 10) -> Dict:
        """
        Classify using EXTREME overfitting with advanced hyperparameters
        """
        
        # Get query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in memory index
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        # Get top matches
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.training_texts):
                # Apply similarity boost
                boosted_sim = min(1.0, sim * self.similarity_boost_factor)
                results.append({
                    'text': self.training_texts[idx],
                    'category': self.training_labels[idx],
                    'similarity': float(boosted_sim),
                    'original_similarity': float(sim),
                    'rank': i + 1
                })
        
        if not results:
            return {
                'domain': 'out_of_scope',
                'category': 'unknown',
                'confidence': 0.99,
                'method': 'no_training_matches',
                'reasoning': 'No training data matches found',
                'top_similarities': []
            }
        
        # Advanced similarity analysis
        if self.use_top_k_averaging:
            # Use top-K averaging instead of just best match
            top_k_results = results[:self.top_k]
            avg_similarity = sum(r['similarity'] for r in top_k_results) / len(top_k_results)
            
            # Category consensus check
            categories = [r['category'] for r in top_k_results]
            category_counts = Counter(categories)
            most_common_category, count = category_counts.most_common(1)[0]
            category_consensus = count / len(top_k_results)
            
            decision_similarity = avg_similarity
            decision_method = f"top_{len(top_k_results)}_avg"
        else:
            # Use single best match
            best_match = results[0]
            decision_similarity = best_match['similarity']
            most_common_category = best_match['category']
            category_consensus = 1.0
            decision_method = "single_best"
        
        # EXTREME threshold checking with multiple levels
        if decision_similarity >= self.ultra_extreme_threshold:
            # Nearly identical - ultra high confidence
            domain = 'namjari'
            confidence = min(0.999, decision_similarity)
            threshold_level = "ultra_extreme"
            reasoning = f"Ultra-extreme similarity {decision_similarity:.4f} >= {self.ultra_extreme_threshold}"
            
        elif decision_similarity >= self.extreme_threshold:
            # Almost identical - very high confidence  
            domain = 'namjari'
            confidence = min(0.95, decision_similarity * 0.9)
            threshold_level = "extreme"
            reasoning = f"Extreme similarity {decision_similarity:.4f} >= {self.extreme_threshold}"
            
        elif decision_similarity >= self.very_high_threshold:
            # Very high similarity - high confidence
            domain = 'namjari'
            confidence = min(0.85, decision_similarity * 0.8)
            threshold_level = "very_high"
            reasoning = f"Very high similarity {decision_similarity:.4f} >= {self.very_high_threshold}"
            
        elif decision_similarity >= self.high_threshold:
            # High similarity - medium confidence
            # Additional check: require category consensus for this level
            if self.require_category_consensus and category_consensus < 0.6:
                domain = 'out_of_scope'
                confidence = 1.0 - decision_similarity * 0.7
                threshold_level = "failed_consensus"
                reasoning = f"High similarity {decision_similarity:.4f} but low consensus {category_consensus:.2f}"
            else:
                domain = 'namjari'
                confidence = min(0.75, decision_similarity * 0.7)
                threshold_level = "high"
                reasoning = f"High similarity {decision_similarity:.4f} >= {self.high_threshold}"
        else:
            # Below high threshold - reject as out-of-scope
            domain = 'out_of_scope'
            confidence = min(0.99, 1.0 - decision_similarity * 0.5)
            threshold_level = "rejected"
            reasoning = f"Low similarity {decision_similarity:.4f} < {self.high_threshold} - REJECTED"
        
        return {
            'domain': domain,
            'category': most_common_category if domain == 'namjari' else 'unknown',
            'confidence': confidence,
            'method': decision_method,
            'threshold_level': threshold_level,
            'reasoning': reasoning,
            'decision_similarity': decision_similarity,
            'category_consensus': category_consensus,
            'top_similarities': [r['similarity'] for r in results[:5]],
            'top_matches': [(r['similarity'], r['category'], r['text'][:100]) for r in results[:3]]
        }
    
    def test_on_user_examples(self):
        """Test the EXTREME overfitting on user's examples"""
        
        user_examples = [
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
        
        # Expected classifications (based on real meaning)
        expected_results = {
            # Land occupation (NOT Namjari - illegal occupation)
            "জমি দখলের কিভাবে পেতে পারি?": "out_of_scope",
            "জমি দখল করতে কি ভূমি অফিসে যাওয়া লাগে?": "out_of_scope",
            "কারো সাহায্যে কি দখল করা যাবে?": "out_of_scope",
            "জমি দখলের জন্য নিবন্ধন করতে কি লাগে?": "out_of_scope",  # Still about illegal occupation
            
            # Clear out-of-scope
            "আমার হারিয়ে যাওয়া বই পেতে কি করতে হবে?": "out_of_scope",
            "হজ্ব করতে চাই, কি করতে হবে?": "out_of_scope", 
            "অনলাইনে ওমরাহ্‌ নিবন্ধন করতে কি করতে হবে?": "out_of_scope",
            "আমি ওমরাহ্‌ করবো কি করতে হবে?": "out_of_scope",
            "কোম্পানিতে আবেদন কি নিজে করতে পারি।": "out_of_scope",
            "আমি নিজে কি চাকরির আবেদন করতে পারবো?": "out_of_scope",
            
            # Birth registration (clear out-of-scope)
            "জন্মনিবন্ধন করতে কি কি দরকার?": "out_of_scope",
            "জন্মনিবন্ধন করতে কি দলিল লাগে?": "out_of_scope", 
            "জন্মনিবন্ধন করতে নিজের মোবাইল থাকতে হবে?": "out_of_scope",
            "জন্মনিবন্ধন করতে মোবাইল নম্বর ছাড়া আর কি লাগে?": "out_of_scope",
            "জন্মনিবন্ধন করতে এনআইডি বা জন্মনিবন্ধন লাগে কি?": "out_of_scope",
            
            # Religious services (out-of-scope)
            "হজ্ব আবেদন নিজে না করলে প্রতিনিধি দিয়ে করা যায় কিনা?": "out_of_scope",
            
            # Other out-of-scope
            "আমি নিজ এলাকার বাইরে থাকি অন্য কেউ কি আমার নামে চাঁদাবাজি করতে পারে কিনা?": "out_of_scope",
            "আমি বিদেশে থাকি আমার ভাই বা আত্মীয় দশ পারসেন্ট আবেদন করতে পারে কিনা?": "out_of_scope",
            
            # Potentially Namjari (could be about muting/mutation)
            "মিউট করতে হলে কি করতে হবে?": "namjari",  # Could be mutation
            
            # Potentially Namjari (could be about land services)
            "আমি নারী জমির কিছুই বুঝি না, আমি সিনেসিস এ আবেদন করাতে পারবো তো?": "namjari", # About land services
        }
        
        print("\n🔥 EXTREME PURE OVERFITTING TEST RESULTS")
        print("="*80)
        
        correct_predictions = 0
        total_predictions = 0
        namjari_correct = 0
        namjari_total = 0
        out_of_scope_correct = 0 
        out_of_scope_total = 0
        
        for query in user_examples:
            result = self.classify_with_extreme_overfitting(query)
            expected = expected_results.get(query, "unknown")
            
            if expected != "unknown":
                total_predictions += 1
                is_correct = result['domain'] == expected
                if is_correct:
                    correct_predictions += 1
                    
                if expected == 'namjari':
                    namjari_total += 1
                    if is_correct:
                        namjari_correct += 1
                elif expected == 'out_of_scope':
                    out_of_scope_total += 1
                    if is_correct:
                        out_of_scope_correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"\n{status} Query: {query}")
                print(f"   🎯 Predicted: {result['domain']} ({result['threshold_level']}, conf: {result['confidence']:.3f})")
                print(f"   📝 Expected: {expected}")
                print(f"   🔍 Method: {result['method']}, Similarity: {result['decision_similarity']:.4f}")
                print(f"   💭 Reasoning: {result['reasoning']}")
            
        # Calculate metrics
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        namjari_accuracy = namjari_correct / namjari_total if namjari_total > 0 else 0
        out_of_scope_accuracy = out_of_scope_correct / out_of_scope_total if out_of_scope_total > 0 else 0
        
        print(f"\n📊 EXTREME OVERFITTING RESULTS:")
        print(f"   🎯 Overall Accuracy: {correct_predictions}/{total_predictions} ({overall_accuracy*100:.1f}%)")
        print(f"   ✅ Namjari Accuracy: {namjari_correct}/{namjari_total} ({namjari_accuracy*100:.1f}%)")
        print(f"   🚫 Out-of-Scope Accuracy: {out_of_scope_correct}/{out_of_scope_total} ({out_of_scope_accuracy*100:.1f}%)")
        
        return {
            'overall_accuracy': overall_accuracy,
            'namjari_accuracy': namjari_accuracy, 
            'out_of_scope_accuracy': out_of_scope_accuracy,
            'total_predictions': total_predictions
        }

def main():
    """Test EXTREME Pure Overfitting"""
    logger.info("🚀 TESTING EXTREME PURE OVERFITTING")
    logger.info("="*80)
    
    classifier = ExtremePureOverfittingClassifier()
    
    # Load and index training data
    classifier.load_all_training_data()
    classifier.create_memory_index()
    
    # Test on user examples
    results = classifier.test_on_user_examples()
    
    print(f"\n🔥 EXTREME OVERFITTING STRATEGY:")
    print(f"   1. ✅ Ultra-Extreme threshold: {classifier.ultra_extreme_threshold}")  
    print(f"   2. ✅ Extreme threshold: {classifier.extreme_threshold}")
    print(f"   3. ✅ Top-{classifier.top_k} similarity averaging")
    print(f"   4. ✅ Category consensus checking")
    print(f"   5. ✅ Similarity boosting factor: {classifier.similarity_boost_factor}")
    print(f"   6. ✅ Multi-level threshold rejection")
    print("="*80)

if __name__ == "__main__":
    main()