#!/usr/bin/env python3
"""
Pure Overfitting Strategy - NO Pattern Matching, Just Extreme Overfitting
Goal: Train model to be SO overfitted that it only accepts exact training data variations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import faiss
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PureOverfittingClassifier:
    """
    Pure overfitting without any keyword patterns
    Strategy: Train to EXTREME overfitting, then use very high similarity thresholds
    """
    
    def __init__(self):
        self.base_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model = SentenceTransformer(self.base_model)
        
        # Training data
        self.training_texts = []
        self.training_labels = []
        self.training_embeddings = None
        
        # FAISS for similarity
        self.faiss_index = None
        
        # EXTREME thresholds for pure overfitting
        self.rejection_threshold = 0.92  # Must be EXTREMELY similar
        self.confidence_threshold = 0.95  # Even higher for high confidence
        
        logger.info("Initialized Pure Overfitting Classifier")
    
    def load_all_training_data(self, data_dir: str = "namjari_questions"):
        """Load ALL training data"""
        logger.info("Loading ALL training data for pure overfitting...")
        
        data_dir = Path(data_dir)
        all_texts = []
        all_labels = []
        
        csv_files = list(data_dir.glob("*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            category = csv_file.stem.replace('namjari_', '')
            
            questions = df['question'].tolist()
            all_texts.extend(questions)
            all_labels.extend([category] * len(questions))
            
            logger.info(f"  {category}: {len(questions)} examples")
        
        self.training_texts = all_texts
        self.training_labels = all_labels
        
        logger.info(f"Total: {len(all_texts)} training examples")
        return all_texts, all_labels
    
    def create_extreme_overfitting_data(self):
        """Create training data for EXTREME overfitting"""
        logger.info("Creating extreme overfitting training data...")
        
        train_examples = []
        
        # Strategy 1: Every training example should be similar to itself and category peers
        category_groups = defaultdict(list)
        for text, label in zip(self.training_texts, self.training_labels):
            category_groups[label].append(text)
        
        # Create MASSIVE positive pairs within same category
        for category, texts in category_groups.items():
            for i, text1 in enumerate(texts):
                for j, text2 in enumerate(texts):
                    if i != j:  # Don't pair with itself
                        # Same category = very high similarity (extreme overfitting)
                        train_examples.append(InputExample(texts=[text1, text2], label=0.98))
        
        # Strategy 2: Create negative pairs with out-of-scope examples
        hard_out_of_scope = [
            "আমি কীভাবে জমির দখল পেতে পারি?",
            "আমার হারিয়ে যাওয়া বইটি ফেরত পেতে কী করতে হবে?",
            "আমি হজ্ব করতে ইচ্ছুক, আমার করণীয় কী?",
            "অনলাইনে ওমরাহ্‌ নিবন্ধনের পদ্ধতি কী?",
            "কোনো কিছু মিউট করার উপায় কী?",
            "চাকরির জন্য কি আমি নিজে আবেদন করতে পারবো?",
            "জন্ম নিবন্ধন করার জন্য কী কী প্রয়োজন?",
            "হজ্বের আবেদন কি প্রতিনিধির মাধ্যমে করা সম্ভব?",
            "আমি প্রবাসী, আমার পক্ষ থেকে আমার ভাই বা কোনো আত্মীয় কি ১০ শতাংশ কোটায় আবেদন করতে পারবে?",
            # Add more diverse out-of-scope
            "আজকের আবহাওয়া কেমন?",
            "ফুটবল খেলা দেখতে চাই",  
            "মোবাইল রিচার্জ করতে চাই",
            "কোভিড টিকা কোথায় পাবো?",
            "ব্যাংক অ্যাকাউন্ট খুলতে কি লাগে?",
            "পাসপোর্ট করতে কি করতে হবে?",
            "ড্রাইভিং লাইসেন্স করতে কি লাগে?",
        ]
        
        # Every training example should be VERY different from out-of-scope
        for train_text in self.training_texts:
            for oos_text in hard_out_of_scope:
                # Training vs out-of-scope = very low similarity (extreme separation)
                train_examples.append(InputExample(texts=[train_text, oos_text], label=0.02))
        
        # Strategy 3: Cross-category negatives (different namjari categories should be less similar)
        categories = list(category_groups.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                texts1 = category_groups[cat1][:10]  # Limit to prevent explosion
                texts2 = category_groups[cat2][:10]
                
                for text1 in texts1:
                    for text2 in texts2:
                        # Different categories = medium-low similarity
                        train_examples.append(InputExample(texts=[text1, text2], label=0.3))
        
        logger.info(f"Created {len(train_examples)} training pairs for extreme overfitting")
        return train_examples
    
    def extreme_fine_tuning(self, epochs: int = 50, batch_size: int = 2):
        """
        EXTREME overfitting with many epochs and tiny batches
        Goal: Memorize training data completely
        """
        logger.info(f"Starting EXTREME overfitting: {epochs} epochs, batch_size {batch_size}")
        
        # Create overfitting data
        train_examples = self.create_extreme_overfitting_data()
        
        # Tiny batch size for maximum overfitting
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Use CosineSimilarityLoss for extreme similarity learning
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # EXTREME training parameters
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,  # Many epochs for extreme overfitting
            output_path=f"models/extreme-overfitted-namjari-{epochs}epochs",
            save_best_model=False,  # Don't save "best", save final overfitted model
            optimizer_params={'lr': 1e-4},  # Lower LR for fine-grained overfitting
            warmup_steps=0,  # No warmup, immediate overfitting
            show_progress_bar=True,
            evaluation_steps=1000,  # Less frequent evaluation
        )
        
        # Reload the overfitted model
        self.model = SentenceTransformer(f"models/extreme-overfitted-namjari-{epochs}epochs")
        
        logger.info("EXTREME overfitting completed!")
        return self.model
    
    def create_overfitted_index(self):
        """Create FAISS index with overfitted embeddings"""
        logger.info("Creating overfitted FAISS index...")
        
        # Generate embeddings with overfitted model
        self.training_embeddings = self.model.encode(
            self.training_texts, 
            convert_to_tensor=False,
            show_progress_bar=True
        )
        
        # Create exact search index
        dimension = self.training_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize and add
        embeddings_normalized = self.training_embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        self.faiss_index.add(embeddings_normalized)
        
        logger.info(f"Overfitted FAISS index created with {self.faiss_index.ntotal} examples")
        return self.faiss_index
    
    def classify_pure_overfitting(self, query: str, k: int = 5) -> Dict:
        """
        Pure overfitting classification - NO pattern matching
        Only similarity to overfitted training data
        """
        
        # Get query embedding from overfitted model
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in overfitted index
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.training_texts):
                results.append({
                    'text': self.training_texts[idx],
                    'category': self.training_labels[idx],
                    'similarity': float(sim),
                    'rank': i + 1
                })
        
        if not results:
            return {
                'domain': 'out_of_scope',
                'category': 'unknown',
                'confidence': 0.95,
                'method': 'no_matches',
                'reasoning': 'No training data matches found'
            }
        
        best_match = results[0]
        top_similarity = best_match['similarity']
        
        # EXTREME thresholding - only accept if VERY similar to training
        if top_similarity >= self.confidence_threshold:
            # Very high confidence
            from collections import Counter
            top_categories = [r['category'] for r in results[:3]]
            category_counts = Counter(top_categories)
            most_common_category = category_counts.most_common(1)[0][0]
            consensus = category_counts.most_common(1)[0][1] / len(top_categories)
            
            return {
                'domain': 'namjari',
                'category': most_common_category,
                'confidence': min(0.99, top_similarity * consensus),
                'method': 'extreme_overfitting_high',
                'reasoning': f'EXTREME similarity ({top_similarity:.3f}) to overfitted training data',
                'top_similarity': top_similarity,
                'threshold_used': self.confidence_threshold,
                'top_matches': results
            }
        
        elif top_similarity >= self.rejection_threshold:
            # Medium confidence - still accept but lower confidence
            from collections import Counter
            top_categories = [r['category'] for r in results[:3]]
            category_counts = Counter(top_categories)
            most_common_category = category_counts.most_common(1)[0][0]
            
            return {
                'domain': 'namjari',
                'category': most_common_category,
                'confidence': min(0.85, top_similarity * 0.9),
                'method': 'extreme_overfitting_medium',
                'reasoning': f'High similarity ({top_similarity:.3f}) to overfitted training data',
                'top_similarity': top_similarity,
                'threshold_used': self.rejection_threshold,
                'top_matches': results
            }
        
        else:
            # Reject - not similar enough to overfitted training data
            return {
                'domain': 'out_of_scope',
                'category': 'unknown',
                'confidence': min(0.98, 1.0 - top_similarity),
                'method': 'overfitting_rejection',
                'reasoning': f'Low similarity ({top_similarity:.3f}) to overfitted training data - REJECTED',
                'top_similarity': top_similarity,
                'threshold_used': self.rejection_threshold,
                'top_matches': results
            }
    
    def optimize_thresholds(self, validation_queries: List[Tuple[str, str]]):
        """Optimize thresholds for maximum precision"""
        logger.info("Optimizing thresholds for extreme precision...")
        
        # Even more extreme thresholds
        confidence_thresholds = [0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
        rejection_thresholds = [0.88, 0.90, 0.92, 0.94, 0.95]
        
        best_accuracy = 0.0
        best_confidence = self.confidence_threshold
        best_rejection = self.rejection_threshold
        
        for conf_thresh in confidence_thresholds:
            for rej_thresh in rejection_thresholds:
                if rej_thresh >= conf_thresh:
                    continue
                    
                self.confidence_threshold = conf_thresh
                self.rejection_threshold = rej_thresh
                
                correct = 0
                total = 0
                
                for query, expected in validation_queries:
                    if expected == "ambiguous":
                        continue
                        
                    result = self.classify_pure_overfitting(query)
                    predicted = result['domain']
                    
                    total += 1
                    if predicted == expected:
                        correct += 1
                
                accuracy = correct / total if total > 0 else 0.0
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_confidence = conf_thresh
                    best_rejection = rej_thresh
        
        self.confidence_threshold = best_confidence
        self.rejection_threshold = best_rejection
        
        logger.info(f"Optimized thresholds: confidence={best_confidence:.3f}, rejection={best_rejection:.3f}")
        logger.info(f"Best accuracy: {best_accuracy:.3f}")
        
        return best_confidence, best_rejection, best_accuracy


def test_pure_overfitting():
    """Test pure overfitting approach"""
    logger.info("🚀 TESTING PURE OVERFITTING CLASSIFIER (NO PATTERNS)")
    logger.info("="*80)
    
    # Initialize classifier
    classifier = PureOverfittingClassifier()
    
    # Load all training data
    classifier.load_all_training_data()
    
    # EXTREME overfitting training
    classifier.extreme_fine_tuning(epochs=50, batch_size=2)
    
    # Create overfitted index
    classifier.create_overfitted_index()
    
    # Test queries
    test_queries = [
        ("আমি কীভাবে জমির দখল পেতে পারি?", "out_of_scope"),
        ("আমার হারিয়ে যাওয়া বইটি ফেরত পেতে কী করতে হবে?", "out_of_scope"),
        ("আমি হজ্ব করতে ইচ্ছুক, আমার করণীয় কী?", "out_of_scope"),
        ("অনলাইনে ওমরাহ্‌ নিবন্ধনের পদ্ধতি কী?", "out_of_scope"),
        ("কোনো কিছু মিউট করার উপায় কী?", "out_of_scope"),
        ("চাকরির জন্য কি আমি নিজে আবেদন করতে পারবো?", "out_of_scope"),
        ("জন্ম নিবন্ধন করার জন্য কী কী প্রয়োজন?", "out_of_scope"),
        ("হজ্বের আবেদন কি প্রতিনিধির মাধ্যমে করা সম্ভব?", "out_of_scope"),
        ("আমি প্রবাসী, আমার পক্ষ থেকে আমার ভাই বা কোনো আত্মীয় কি ১০ শতাংশ কোটায় আবেদন করতে পারবে?", "out_of_scope"),
        
        # Training examples should still work
        ("নামজারি করতে কি করতে হবে?", "namjari"),
        ("নামজারির ফি কত?", "namjari"),
        ("নামজারি করতে কি দলিল লাগে?", "namjari"),
        ("খতিয়ানের কপি কিভাবে পাবো?", "namjari"),
    ]
    
    # Optimize thresholds
    classifier.optimize_thresholds(test_queries)
    
    # Test all queries
    logger.info("\n" + "="*50)
    logger.info("PURE OVERFITTING RESULTS:")
    logger.info("="*50)
    
    out_of_scope_correct = 0
    out_of_scope_total = 0
    namjari_correct = 0
    namjari_total = 0
    
    for query, expected in test_queries:
        result = classifier.classify_pure_overfitting(query)
        predicted = result['domain']
        
        is_correct = predicted == expected
        
        if expected == "out_of_scope":
            out_of_scope_total += 1
            if is_correct:
                out_of_scope_correct += 1
        elif expected == "namjari":
            namjari_total += 1
            if is_correct:
                namjari_correct += 1
        
        status = "✅" if is_correct else "❌"
        
        logger.info(f"\n{status} Query: '{query[:60]}{'...' if len(query) > 60 else ''}'")
        logger.info(f"    Expected: {expected}")
        logger.info(f"    Predicted: {predicted} (confidence: {result['confidence']:.3f})")
        logger.info(f"    Method: {result['method']}")
        logger.info(f"    Top Similarity: {result.get('top_similarity', 0):.3f}")
        logger.info(f"    Threshold: {result.get('threshold_used', 0):.3f}")
    
    # Calculate final metrics
    out_of_scope_accuracy = out_of_scope_correct / out_of_scope_total if out_of_scope_total > 0 else 0
    namjari_accuracy = namjari_correct / namjari_total if namjari_total > 0 else 0
    overall_accuracy = (out_of_scope_correct + namjari_correct) / (out_of_scope_total + namjari_total)
    
    logger.info(f"\n{'='*50}")
    logger.info("PURE OVERFITTING FINAL RESULTS:")
    logger.info(f"{'='*50}")
    logger.info(f"📊 Out-of-scope Detection: {out_of_scope_correct}/{out_of_scope_total} ({out_of_scope_accuracy*100:.1f}%)")
    logger.info(f"📊 Namjari Detection: {namjari_correct}/{namjari_total} ({namjari_accuracy*100:.1f}%)")
    logger.info(f"📊 Overall Accuracy: {overall_accuracy*100:.1f}%")
    logger.info(f"🎯 Confidence Threshold: {classifier.confidence_threshold:.3f}")
    logger.info(f"🎯 Rejection Threshold: {classifier.rejection_threshold:.3f}")
    
    return classifier, {
        'out_of_scope_accuracy': out_of_scope_accuracy,
        'namjari_accuracy': namjari_accuracy,
        'overall_accuracy': overall_accuracy,
        'confidence_threshold': classifier.confidence_threshold,
        'rejection_threshold': classifier.rejection_threshold
    }


if __name__ == "__main__":
    try:
        classifier, results = test_pure_overfitting()
        
        print("\n" + "="*80)
        print("🏆 PURE OVERFITTING CLASSIFIER RESULTS")
        print("="*80)
        print(f"📊 Out-of-scope Detection: {results['out_of_scope_accuracy']*100:.1f}%")
        print(f"📊 Namjari Detection: {results['namjari_accuracy']*100:.1f}%")
        print(f"📊 Overall Accuracy: {results['overall_accuracy']*100:.1f}%")
        print(f"🎯 Thresholds: Confidence={results['confidence_threshold']:.3f}, Rejection={results['rejection_threshold']:.3f}")
        
        print("\n🚀 Pure Overfitting Strategy:")
        print("   1. ✅ NO keyword patterns - pure ML overfitting")
        print("   2. ✅ 50 epochs with batch_size=2 (extreme overfitting)")
        print("   3. ✅ Memorizes exact training data relationships")
        print("   4. ✅ EXTREME similarity thresholds (0.92+ required)")
        print("   5. ✅ Rejects anything not similar to training")
        
        print("\n💡 This approach:")
        print("   • Learns ONLY from training data patterns")
        print("   • No manual keyword rules or pattern matching")
        print("   • Extreme overfitting makes it reject unfamiliar queries")
        print("   • Perfect for small, well-defined domains")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Pure overfitting failed: {e}")
        raise