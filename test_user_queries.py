#!/usr/bin/env python3
"""
Comprehensive Test Script for User's Bengali Queries
Tests both production systems with the provided Bengali questions
"""

import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from namjari_query_handler import NamjariQueryHandler
from production_intent_system import ProductionIntentSystem
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveQueryTester:
    """
    Test suite for user's Bengali queries across different systems
    """
    
    def __init__(self):
        self.user_queries = [
            "আমি কীভাবে জমির দখল পেতে পারি?",
            "আমার হারিয়ে যাওয়া বইটি ফেরত পেতে কী করতে হবে?",
            "আমি হজ্ব করতে ইচ্ছুক, আমার করণীয় কী?",
            "অনলাইনে ওমরাহ্‌ নিবন্ধনের পদ্ধতি কী?",
            "কোনো কিছু মিউট করার উপায় কী?",
            "আমি ওমরাহ্‌ করতে চাই, এর জন্য কী করতে হবে?",
            "আমি কি কোনো কোম্পানিতে সরাসরি চাকরির আবেদন করতে পারি?",
            "জমির দখল পেতে হলে ভূমি অফিসে যেতে হয় কি?",
            "চাকরির জন্য কি আমি নিজে আবেদন করতে পারবো?",
            "অন্য কারো সহায়তায় কি জমির দখল নেওয়া যায়?",
            "জমির দখল বা মালিকানা নিবন্ধনের জন্য কী কী প্রয়োজন?",
            "জন্ম নিবন্ধন করার জন্য কী কী প্রয়োজন?",
            "জন্ম নিবন্ধন করার জন্য কি কোনো দলিলের প্রয়োজন হয়?",
            "জন্ম নিবন্ধনের জন্য আবেদনকারীর কি নিজের মোবাইল নম্বর থাকা আবশ্যক?",
            "জন্ম নিবন্ধনের জন্য মোবাইল নম্বর ছাড়া আর কী কী প্রয়োজন?",
            "সন্তানের জন্ম নিবন্ধনের জন্য কি বাবা-মায়ের জাতীয় পরিচয়পত্র বা জন্ম নিবন্ধন সনদ প্রয়োজন?",
            "হজ্বের আবেদন কি প্রতিনিধির মাধ্যমে করা সম্ভব?",
            "আমি এলাকায় না থাকার সুযোগে, অন্য কেউ কি আমার নাম ব্যবহার করে চাঁদাবাজি করতে পারে?",
            "আমি প্রবাসী, আমার পক্ষ থেকে আমার ভাই বা কোনো আত্মীয় কি ১০ শতাংশ কোটায় আবেদন করতে পারবে?",
        ]
        
        # Expected classifications (manual annotation)
        self.expected_classifications = {
            # Land occupation/possession (NOT Namjari - this is about illegal occupation)
            "আমি কীভাবে জমির দখল পেতে পারি?": "out_of_scope",
            "জমির দখল পেতে হলে ভূমি অফিসে যেতে হয় কি?": "out_of_scope", 
            "অন্য কারো সহায়তায় কি জমির দখল নেওয়া যায়?": "out_of_scope",
            
            # Ambiguous - could be Namjari (ownership transfer) or land grab
            "জমির দখল বা মালিকানা নিবন্ধনের জন্য কী কী প্রয়োজন?": "ambiguous_namjari",
            
            # Clear out-of-scope
            "আমার হারিয়ে যাওয়া বইটি ফেরত পেতে কী করতে হবে?": "out_of_scope",
            "আমি হজ্ব করতে ইচ্ছুক, আমার করণীয় কী?": "out_of_scope",
            "অনলাইনে ওমরাহ্‌ নিবন্ধনের পদ্ধতি কী?": "out_of_scope",
            "কোনো কিছু মিউট করার উপায় কী?": "out_of_scope",
            "আমি ওমরাহ্‌ করতে চাই, এর জন্য কী করতে হবে?": "out_of_scope",
            "আমি কি কোনো কোম্পানিতে সরাসরি চাকরির আবেদন করতে পারি?": "out_of_scope",
            "চাকরির জন্য কি আমি নিজে আবেদন করতে পারবো?": "out_of_scope",
            "হজ্বের আবেদন কি প্রতিনিধির মাধ্যমে করা সম্ভব?": "out_of_scope",
            
            # Birth registration (clear out-of-scope)
            "জন্ম নিবন্ধন করার জন্য কী কী প্রয়োজন?": "out_of_scope",
            "জন্ম নিবন্ধন করার জন্য কি কোনো দলিলের প্রয়োজন হয়?": "out_of_scope",
            "জন্ম নিবন্ধনের জন্য আবেদনকারীর কি নিজের মোবাইল নম্বর থাকা আবশ্যক?": "out_of_scope",
            "জন্ম নিবন্ধনের জন্য মোবাইল নম্বর ছাড়া আর কী কী প্রয়োজন?": "out_of_scope",
            "সন্তানের জন্ম নিবন্ধনের জন্য কি বাবা-মায়ের জাতীয় পরিচয়পত্র বা জন্ম নিবন্ধন সনদ প্রয়োজন?": "out_of_scope",
            
            # Other out-of-scope
            "আমি এলাকায় না থাকার সুযোগে, অন্য কেউ কি আমার নাম ব্যবহার করে চাঁদাবাজি করতে পারে?": "out_of_scope",
            "আমি প্রবাসী, আমার পক্ষ থেকে আমার ভাই বা কোনো আত্মীয় কি ১০ শতাংশ কোটায় আবেদন করতে পারবে?": "out_of_scope",
        }
        
    def test_query_handler(self):
        """Test with NamjariQueryHandler"""
        logger.info("=== Testing with NamjariQueryHandler ===")
        
        handler = NamjariQueryHandler()
        results = []
        
        for query in self.user_queries:
            result = handler.handle_query(query)
            results.append((query, result))
            
            # Get expected classification
            expected = self.expected_classifications.get(query, "unknown")
            is_correct = "unknown"
            
            if expected == "out_of_scope":
                is_correct = "✅" if result['domain'] == 'out_of_scope' else "❌"
            elif expected == "ambiguous_namjari":
                is_correct = "🤔" if result['domain'] == 'namjari' else "❌"
            
            logger.info(f"\nQuery: {query}")
            logger.info(f"  Result: {result['domain']} (confidence: {result['domain_confidence']:.3f})")
            logger.info(f"  Expected: {expected} {is_correct}")
            logger.info(f"  Reasoning: {result['domain_reasoning']}")
            if result['category']:
                logger.info(f"  Category: {result['category']} (confidence: {result['category_confidence']:.3f})")
                
        return results
    
    def test_production_system(self):
        """Test with Production Intent Classification System"""
        logger.info("\n=== Testing with Production Intent System ===")
        
        try:
            # Load binary classifier
            binary_model = AutoModelForSequenceClassification.from_pretrained("models/binary-classifier/final")
            binary_tokenizer = AutoTokenizer.from_pretrained("models/binary-classifier/final")
            logger.info("✅ Binary classifier loaded")
        except Exception as e:
            logger.warning(f"Binary classifier not found: {e}")
            logger.info("Please run production_intent_system.py first to train the model")
            return []
        
        results = []
        
        for query in self.user_queries:
            # Binary classification
            inputs = binary_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64)
            
            with torch.no_grad():
                outputs = binary_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                namjari_probability = predictions[0][1].item()
                is_classified_namjari = namjari_probability > 0.5
            
            # Get expected classification
            expected = self.expected_classifications.get(query, "unknown")
            is_correct = "unknown"
            
            if expected == "out_of_scope":
                is_correct = "✅" if not is_classified_namjari else "❌"
            elif expected == "ambiguous_namjari":
                is_correct = "🤔" if is_classified_namjari else "❌"
            
            result = {
                'domain': 'namjari' if is_classified_namjari else 'out_of_scope',
                'confidence': namjari_probability if is_classified_namjari else 1-namjari_probability,
                'namjari_probability': namjari_probability
            }
            
            results.append((query, result))
            
            logger.info(f"\nQuery: {query}")
            logger.info(f"  Result: {result['domain']} (namjari_prob: {namjari_probability:.3f})")
            logger.info(f"  Expected: {expected} {is_correct}")
            
        return results
    
    def analyze_performance(self, handler_results, production_results):
        """Analyze and compare performance"""
        logger.info("\n=== PERFORMANCE ANALYSIS ===")
        
        # Count correct classifications
        handler_correct = 0
        production_correct = 0
        total_clear_cases = 0
        
        for query in self.user_queries:
            expected = self.expected_classifications.get(query, "unknown")
            if expected == "unknown" or expected == "ambiguous_namjari":
                continue
                
            total_clear_cases += 1
            
            # Find results
            handler_result = next((r for q, r in handler_results if q == query), None)
            production_result = next((r for q, r in production_results if q == query), None)
            
            if handler_result:
                if expected == "out_of_scope" and handler_result['domain'] == 'out_of_scope':
                    handler_correct += 1
                elif expected == "namjari" and handler_result['domain'] == 'namjari':
                    handler_correct += 1
            
            if production_result:
                if expected == "out_of_scope" and production_result['domain'] == 'out_of_scope':
                    production_correct += 1
                elif expected == "namjari" and production_result['domain'] == 'namjari':
                    production_correct += 1
        
        handler_accuracy = handler_correct / total_clear_cases if total_clear_cases > 0 else 0
        production_accuracy = production_correct / total_clear_cases if total_clear_cases > 0 else 0
        
        logger.info(f"Query Handler Accuracy: {handler_correct}/{total_clear_cases} ({handler_accuracy*100:.1f}%)")
        logger.info(f"Production System Accuracy: {production_correct}/{total_clear_cases} ({production_accuracy*100:.1f}%)")
        
        # Identify problem cases
        logger.info("\n=== PROBLEM CASES ===")
        
        for query in self.user_queries:
            expected = self.expected_classifications.get(query, "unknown")
            if expected == "unknown":
                continue
                
            handler_result = next((r for q, r in handler_results if q == query), None)
            production_result = next((r for q, r in production_results if q == query), None)
            
            handler_wrong = False
            production_wrong = False
            
            if handler_result:
                if expected == "out_of_scope" and handler_result['domain'] != 'out_of_scope':
                    handler_wrong = True
                elif expected == "namjari" and handler_result['domain'] != 'namjari':
                    handler_wrong = True
            
            if production_result:
                if expected == "out_of_scope" and production_result['domain'] != 'out_of_scope':
                    production_wrong = True
                elif expected == "namjari" and production_result['domain'] != 'namjari':
                    production_wrong = True
            
            if handler_wrong or production_wrong:
                logger.info(f"\nProblem: {query}")
                logger.info(f"  Expected: {expected}")
                if handler_wrong:
                    logger.info(f"  Handler: {handler_result['domain']} ❌")
                if production_wrong:
                    logger.info(f"  Production: {production_result['domain']} ❌")
        
        return {
            'handler_accuracy': handler_accuracy,
            'production_accuracy': production_accuracy,
            'total_clear_cases': total_clear_cases
        }

def main():
    """Main test execution"""
    logger.info("🧪 COMPREHENSIVE QUERY TESTING")
    logger.info("="*80)
    
    tester = ComprehensiveQueryTester()
    
    # Test both systems
    handler_results = tester.test_query_handler()
    production_results = tester.test_production_system()
    
    if production_results:  # Only analyze if production system worked
        performance = tester.analyze_performance(handler_results, production_results)
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TEST RESULTS")
        print("="*80)
        print(f"📊 Query Handler Accuracy: {performance['handler_accuracy']*100:.1f}%")
        print(f"📊 Production System Accuracy: {performance['production_accuracy']*100:.1f}%")
        print(f"📊 Total Test Cases: {performance['total_clear_cases']}")
        
        print("\n🔍 Key Findings:")
        print("   • Most queries are clearly out-of-scope (birth registration, hajj, jobs)")
        print("   • 'জমির দখল' queries are tricky - could be confused with Namjari")
        print("   • Syntactic similarity in Bengali makes classification challenging")
        print("   • Keyword-based approach helps with clear indicators")
        
        print("\n📝 Recommendations:")
        print("   1. Add more 'দখল' (occupation) examples to training data as out-of-scope")
        print("   2. Strengthen keyword filtering for religious terms (হজ্ব, ওমরাহ)")
        print("   3. Add birth registration keywords to out-of-scope list")
        print("   4. Consider confidence thresholds for uncertain cases")
        
        print("="*80)
    else:
        logger.info("Production system not available. Train the model first with:")
        logger.info("python production_intent_system.py")

if __name__ == "__main__":
    main()
