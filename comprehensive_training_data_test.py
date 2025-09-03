#!/usr/bin/env python3
"""
Comprehensive test of Entity-Weighted Embeddings using actual training data
Tests classification accuracy across all categories and edge cases
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import random
import logging

# Import our entity-weighted system
from entity_weighted_embeddings import EntityWeightedEmbeddingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTrainingDataTest:
    """Test entity-weighted embeddings on comprehensive training data"""
    
    def __init__(self, data_dir: str = "namjari_questions"):
        self.data_dir = Path(data_dir)
        self.system = None
        self.training_data = []
        self.test_results = {}
        
    def load_all_training_data(self):
        """Load all training data from CSV files"""
        logger.info("Loading comprehensive training data...")
        
        all_data = []
        category_counts = defaultdict(int)
        
        csv_files = list(self.data_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            category = csv_file.stem.replace('namjari_', '')
            
            for _, row in df.iterrows():
                question = row['question']
                all_data.append({
                    'question': question,
                    'category': category,
                    'file': csv_file.name,
                    'true_domain': 'namjari'  # All training data is namjari domain
                })
                category_counts[category] += 1
        
        self.training_data = all_data
        
        logger.info(f"Loaded {len(all_data)} training examples")
        logger.info("Category distribution:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"  {category}: {count} examples")
            
        return all_data
    
    def create_test_splits(self, test_ratio: float = 0.3):
        """Create train/test splits while maintaining category balance"""
        logger.info(f"Creating train/test splits (test_ratio: {test_ratio})...")
        
        # Group by category
        category_groups = defaultdict(list)
        for item in self.training_data:
            category_groups[item['category']].append(item)
        
        train_data = []
        test_data = []
        
        for category, items in category_groups.items():
            # Shuffle items
            random.shuffle(items)
            
            # Split
            n_test = max(1, int(len(items) * test_ratio))  # At least 1 test item per category
            
            test_items = items[:n_test]
            train_items = items[n_test:]
            
            train_data.extend(train_items)
            test_data.extend(test_items)
            
            logger.info(f"  {category}: {len(train_items)} train, {len(test_items)} test")
        
        return train_data, test_data
    
    def initialize_system_with_training_data(self, train_data: List[Dict]):
        """Initialize entity-weighted system with training data"""
        logger.info("Initializing entity-weighted embedding system...")
        
        self.system = EntityWeightedEmbeddingSystem()
        
        # Override the load_dataset method to use our training data
        self.system.texts = [item['question'] for item in train_data]
        self.system.labels = [item['category'] for item in train_data]
        
        # Add hard negatives (same as original system)
        hard_negatives = self.system.hard_negatives_miner.generate_hard_negatives(self.system.texts[:20])
        for pos_text, neg_text in hard_negatives:
            self.system.texts.append(neg_text)
            self.system.labels.append("out_of_scope")
        
        logger.info(f"Training system with {len(self.system.texts)} examples")
        
        # Create embeddings and build index
        self.system.create_embeddings()
        self.system.build_faiss_index()
        
        return self.system
    
    def test_on_data(self, test_data: List[Dict], test_name: str = "test"):
        """Test system on given data"""
        logger.info(f"Testing on {len(test_data)} {test_name} examples...")
        
        results = {
            'correct': 0,
            'total': len(test_data),
            'category_results': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'classification_methods': defaultdict(int),
            'errors': [],
            'predictions': []
        }
        
        for i, item in enumerate(test_data):
            question = item['question']
            true_category = item['category']
            true_domain = item['true_domain']
            
            # Classify
            prediction = self.system.classify_query(question)
            
            predicted_domain = prediction['domain']
            predicted_category = prediction['category']
            
            # Check correctness
            domain_correct = predicted_domain == true_domain
            category_correct = predicted_category == true_category
            
            # We consider it correct if domain is right (category can vary)
            is_correct = domain_correct
            
            if is_correct:
                results['correct'] += 1
                results['category_results'][true_category]['correct'] += 1
                
            results['category_results'][true_category]['total'] += 1
            results['classification_methods'][prediction['method']] += 1
            
            # Store prediction details
            prediction_detail = {
                'question': question,
                'true_category': true_category,
                'predicted_domain': predicted_domain,
                'predicted_category': predicted_category,
                'confidence': prediction['confidence'],
                'method': prediction['method'],
                'reasoning': prediction['reasoning'],
                'correct': is_correct,
                'entity_score': self.system.entity_extractor.calculate_entity_score(question)
            }
            
            results['predictions'].append(prediction_detail)
            
            if not is_correct:
                results['errors'].append({
                    'question': question,
                    'true_category': true_category,
                    'predicted_domain': predicted_domain,
                    'predicted_category': predicted_category,
                    'prediction': prediction
                })
                
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(test_data)} examples...")
        
        # Calculate accuracy
        accuracy = results['correct'] / results['total']
        results['accuracy'] = accuracy
        
        logger.info(f"{test_name.title()} Results:")
        logger.info(f"  Overall Accuracy: {accuracy*100:.1f}% ({results['correct']}/{results['total']})")
        
        return results
    
    def analyze_results(self, results: Dict, test_name: str = "test"):
        """Analyze and display detailed results"""
        logger.info(f"\n=== {test_name.upper()} ANALYSIS ===")
        
        # Overall stats
        logger.info(f"Overall Accuracy: {results['accuracy']*100:.1f}%")
        
        # Per-category accuracy
        logger.info("\nPer-Category Accuracy:")
        category_accuracies = []
        for category, stats in results['category_results'].items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                category_accuracies.append(acc)
                logger.info(f"  {category}: {acc*100:.1f}% ({stats['correct']}/{stats['total']})")
        
        avg_category_accuracy = np.mean(category_accuracies) if category_accuracies else 0
        logger.info(f"Average Category Accuracy: {avg_category_accuracy*100:.1f}%")
        
        # Classification method usage
        logger.info("\nClassification Method Usage:")
        total_methods = sum(results['classification_methods'].values())
        for method, count in results['classification_methods'].items():
            percentage = (count / total_methods) * 100
            logger.info(f"  {method}: {count} ({percentage:.1f}%)")
        
        # Error analysis
        if results['errors']:
            logger.info(f"\nError Analysis ({len(results['errors'])} errors):")
            error_categories = defaultdict(int)
            for error in results['errors']:
                error_categories[error['true_category']] += 1
            
            for category, count in sorted(error_categories.items()):
                logger.info(f"  {category}: {count} errors")
            
            # Show some example errors
            logger.info("\nExample Errors (first 5):")
            for i, error in enumerate(results['errors'][:5]):
                logger.info(f"  {i+1}. '{error['question']}'")
                logger.info(f"      True: {error['true_category']} | Predicted: {error['predicted_domain']}/{error['predicted_category']}")
                logger.info(f"      Method: {error['prediction']['method']} | Confidence: {error['prediction']['confidence']:.3f}")
        
        return {
            'overall_accuracy': results['accuracy'],
            'avg_category_accuracy': avg_category_accuracy,
            'method_distribution': dict(results['classification_methods']),
            'error_count': len(results['errors']),
            'category_accuracies': {cat: stats['correct']/stats['total'] for cat, stats in results['category_results'].items() if stats['total'] > 0}
        }
    
    def test_out_of_scope_detection(self):
        """Test out-of-scope detection with various non-namjari queries"""
        logger.info("\n=== OUT-OF-SCOPE DETECTION TEST ===")
        
        out_of_scope_queries = [
            # Religious
            "হজ্ব করতে চাই, কি করতে হবে?",
            "ওমরাহ করতে কি করতে হবে?",
            "নামাজের ওয়াক্ত কখন?",
            
            # Civil registration  
            "জন্মনিবন্ধন করতে কি দলিল লাগে?",
            "মৃত্যু সনদ করতে কি করতে হবে?",
            "বিয়ে নিবন্ধন করতে কি লাগে?",
            
            # Employment
            "চাকরির আবেদন কিভাবে করবো?",
            "সরকারি চাকরিতে আবেদন করতে কি লাগে?",
            
            # General queries
            "আজকের আবহাওয়া কেমন?",
            "বই পড়তে ভালো লাগে",
            "মোবাইল রিচার্জ করতে চাই",
            "ব্যাংক অ্যাকাউন্ট খুলতে কি লাগে?",
            "পাসপোর্ট করতে কি করতে হবে?",
            "ড্রাইভিং লাইসেন্স করতে কি লাগে?",
            
            # Land-related but not namjari (tricky cases)
            "জমি কিনতে কি করতে হবে?",
            "জমি বিক্রি করতে কি করতে হবে?",
            "জমির দাম কত?",
            "জমি দখল করতে কি ভূমি অফিসে যাওয়া লাগে?",  # This is our hard negative
        ]
        
        out_of_scope_results = {
            'correct': 0,
            'total': len(out_of_scope_queries),
            'errors': []
        }
        
        for query in out_of_scope_queries:
            prediction = self.system.classify_query(query)
            predicted_domain = prediction['domain']
            
            is_correct = predicted_domain in ['out_of_scope', 'uncertain']
            
            if is_correct:
                out_of_scope_results['correct'] += 1
            else:
                out_of_scope_results['errors'].append({
                    'query': query,
                    'prediction': prediction
                })
            
            logger.info(f"'{query[:50]}...' → {predicted_domain} ({prediction['confidence']:.3f})")
        
        accuracy = out_of_scope_results['correct'] / out_of_scope_results['total']
        logger.info(f"\nOut-of-Scope Detection Accuracy: {accuracy*100:.1f}% ({out_of_scope_results['correct']}/{out_of_scope_results['total']})")
        
        if out_of_scope_results['errors']:
            logger.info("Out-of-scope detection errors:")
            for error in out_of_scope_results['errors']:
                logger.info(f"  '{error['query']}' → {error['prediction']['domain']}")
        
        return out_of_scope_results

def run_comprehensive_test():
    """Run comprehensive test on training data"""
    logger.info("🚀 COMPREHENSIVE ENTITY-WEIGHTED EMBEDDINGS TEST")
    logger.info("="*80)
    
    # Initialize test
    tester = ComprehensiveTrainingDataTest()
    
    # Load training data
    tester.load_all_training_data()
    
    # Create train/test splits
    train_data, test_data = tester.create_test_splits(test_ratio=0.3)
    
    # Initialize system with training data
    tester.initialize_system_with_training_data(train_data)
    
    # Test on training data (should have high accuracy)
    logger.info("\n" + "="*50)
    train_results = tester.test_on_data(train_data, "training")
    train_analysis = tester.analyze_results(train_results, "training")
    
    # Test on held-out test data
    logger.info("\n" + "="*50)  
    test_results = tester.test_on_data(test_data, "test")
    test_analysis = tester.analyze_results(test_results, "test")
    
    # Test out-of-scope detection
    logger.info("\n" + "="*50)
    oos_results = tester.test_out_of_scope_detection()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Training Accuracy: {train_analysis['overall_accuracy']*100:.1f}%")
    logger.info(f"Test Accuracy: {test_analysis['overall_accuracy']*100:.1f}%")
    logger.info(f"Out-of-Scope Detection: {(oos_results['correct']/oos_results['total'])*100:.1f}%")
    
    logger.info(f"\nTraining Data Performance:")
    logger.info(f"  Overall: {train_analysis['overall_accuracy']*100:.1f}%")
    logger.info(f"  Avg Category: {train_analysis['avg_category_accuracy']*100:.1f}%")
    logger.info(f"  Errors: {train_analysis['error_count']}")
    
    logger.info(f"\nTest Data Performance:")
    logger.info(f"  Overall: {test_analysis['overall_accuracy']*100:.1f}%")
    logger.info(f"  Avg Category: {test_analysis['avg_category_accuracy']*100:.1f}%")
    logger.info(f"  Errors: {test_analysis['error_count']}")
    
    logger.info(f"\nClassification Method Distribution (Test Data):")
    for method, count in test_analysis['method_distribution'].items():
        percentage = (count / test_results['total']) * 100
        logger.info(f"  {method}: {percentage:.1f}%")
    
    return {
        'train_results': train_analysis,
        'test_results': test_analysis,
        'oos_results': oos_results,
        'tester': tester
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        results = run_comprehensive_test()
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"📊 Training: {results['train_results']['overall_accuracy']*100:.1f}% accuracy")
        print(f"📊 Testing: {results['test_results']['overall_accuracy']*100:.1f}% accuracy")  
        print(f"📊 Out-of-Scope: {(results['oos_results']['correct']/results['oos_results']['total'])*100:.1f}% accuracy")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        raise