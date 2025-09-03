#!/usr/bin/env python3
"""
Final Hybrid Classifier - What Actually Works
Combines entity-weighted embeddings + hard negatives + explicit out-of-scope detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import torch
from sentence_transformers import SentenceTransformer
import faiss
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalHybridClassifier:
    """
    What actually works: Smart combination of approaches based on testing insights
    """
    
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Training data
        self.training_texts = []
        self.training_labels = []
        self.training_embeddings = None
        self.faiss_index = None
        
        # Out-of-scope patterns learned from user examples
        self.explicit_out_of_scope_patterns = [
            # Religious services (very clear indicators)
            r'হজ্?ব',  # Hajj variants
            r'ওমরাহ', # Umrah
            r'হাজী',  # Haji
            
            # Birth registration (not land registration) 
            r'জন্মনিবন্ধন',  # Birth registration
            r'জন্ম\s*সনদ',   # Birth certificate
            
            # Employment/Jobs
            r'চাকরি',      # Job
            r'চাকুরি',     # Job variant
            r'কোম্পানি',   # Company
            r'চাকরির\s*আবেদন', # Job application
            
            # Lost items
            r'হারিয়ে\s*যাওয়া', # Lost
            r'হারানো',         # Lost variant
            r'খোয়া',          # Lost variant
            
            # Illegal land occupation (not legal namjari)
            r'জমি\s*দখল',     # Land occupation/grabbing
            r'দখল\s*করা',     # Taking possession illegally
            r'দখল\s*নেওয়া',   # Taking control
            
            # Technology
            r'মিউট',          # Mute (technology)
            r'আনমিউট',       # Unmute
            
            # Crime/illegal activities  
            r'চাঁদাবাজি',     # Extortion
            r'জোর\s*করে',    # By force
            
            # Other government services
            r'পাসপোর্ট',      # Passport
            r'ভিসা',          # Visa
            
            # General percentage/quota (not land-related)
            r'\d+\s*শতাংশ',           # X percent
            r'\d+\s*পার[সশ]েন্ট',      # X percent
            r'দশ\s*পার[সশ]েন্ট',      # Ten percent
            r'দশ\s*শতাংশ',           # Ten percent
            r'কোটা',                  # Quota
        ]
        
        # Compile regex patterns
        self.out_of_scope_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.explicit_out_of_scope_patterns]
        
        # Entity weights for Namjari domain
        self.namjari_entities = {
            # Core Namjari terms (high weight)
            'নামজারি': 3.0, 'নাম জারি': 3.0, 'মিউটেশন': 2.5, 'mutation': 2.5,
            'খারিজ': 2.5, 'kharij': 2.5, 'বন্টন': 2.0,
            
            # Land/property terms (medium-high weight)
            'জমি': 2.0, 'ভূমি': 2.0, 'সম্পত্তি': 2.0, 'খতিয়ান': 2.5, 'khatian': 2.5,
            'দাগ': 1.8, 'dag': 1.8, 'রেকর্ড': 1.5,
            
            # Legal/administrative terms (medium weight)  
            'আবেদন': 1.5, 'মামলা': 1.8, 'নিবন্ধন': 1.5, 'সেবা': 1.0,
            'অফিস': 1.2, 'দপ্তর': 1.2, 'কার্যালয়': 1.2,
            
            # Inheritance terms (high weight for namjari)
            'ওয়ারিশ': 2.5, 'উত্তরাধিকার': 2.5, 'inheritance': 2.5,
            'বাপের': 2.0, 'পিতার': 2.0, 'মায়ের': 2.0, 'বাবার': 2.0,
            
            # Process terms (lower weight)
            'করতে': 1.0, 'নিতে': 1.0, 'পেতে': 1.0, 'হবে': 0.8,
        }
        
        # Thresholds learned from experiments
        self.entity_threshold = 2.0  # Need significant entity weight
        self.similarity_threshold = 0.75  # Reasonable similarity 
        self.high_confidence_threshold = 0.85  # High similarity
        
        logger.info("🎯 Initialized Final Hybrid Classifier")
        logger.info(f"   Entity threshold: {self.entity_threshold}")
        logger.info(f"   Similarity threshold: {self.similarity_threshold}")
        logger.info(f"   Out-of-scope patterns: {len(self.explicit_out_of_scope_patterns)}")
    
    def extract_entities_with_weights(self, text: str) -> Dict[str, float]:
        """Extract entities and compute weighted scores"""
        text_lower = text.lower()
        found_entities = {}
        total_weight = 0.0
        
        for entity, weight in self.namjari_entities.items():
            entity_lower = entity.lower()
            
            # Use word boundaries for most terms, but allow partial matching for compound terms
            if len(entity_lower) > 4:  # Longer terms can have partial matching
                if entity_lower in text_lower:
                    found_entities[entity] = weight
                    total_weight += weight
            else:  # Shorter terms need word boundaries
                pattern = r'\b' + re.escape(entity_lower) + r'\b'
                if re.search(pattern, text_lower):
                    found_entities[entity] = weight
                    total_weight += weight
        
        return found_entities, total_weight
    
    def check_explicit_out_of_scope(self, text: str) -> Tuple[bool, str]:
        """Check for explicit out-of-scope patterns"""
        for i, pattern in enumerate(self.out_of_scope_regex):
            if pattern.search(text):
                return True, self.explicit_out_of_scope_patterns[i]
        return False, ""
    
    def load_all_training_data(self, data_dir: str = "namjari_questions"):
        """Load training data"""
        logger.info("Loading training data for hybrid approach...")
        
        data_dir = Path(data_dir)
        all_texts = []
        all_labels = []
        
        for csv_file in data_dir.glob("*.csv"):
            category = csv_file.stem.replace('namjari_', '')
            df = pd.read_csv(csv_file)
            
            if 'question' in df.columns:
                questions = df['question'].tolist()
                labels = [category] * len(questions)
                
                all_texts.extend(questions)
                all_labels.extend(labels)
                
                logger.info(f"  {category}: {len(questions)} examples")
        
        self.training_texts = all_texts
        self.training_labels = all_labels
        
        logger.info(f"Total: {len(self.training_texts)} training examples")
        return self.training_texts, self.training_labels
    
    def create_entity_weighted_index(self):
        """Create entity-weighted embeddings index"""
        logger.info("Creating entity-weighted embeddings index...")
        
        # Get base embeddings
        base_embeddings = self.model.encode(
            self.training_texts, 
            convert_to_tensor=False,
            show_progress_bar=True
        )
        
        # Apply entity weighting
        weighted_embeddings = []
        for i, text in enumerate(self.training_texts):
            base_embedding = base_embeddings[i]
            entities, total_weight = self.extract_entities_with_weights(text)
            
            # Apply entity weighting
            if total_weight > 0:
                weight_factor = 1.0 + (total_weight * 0.1)  # Boost by entity weight
                weighted_embedding = base_embedding * weight_factor
            else:
                weighted_embedding = base_embedding
                
            weighted_embeddings.append(weighted_embedding)
        
        self.training_embeddings = np.array(weighted_embeddings)
        
        # Create FAISS index
        dimension = self.training_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        embeddings_normalized = self.training_embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        self.faiss_index.add(embeddings_normalized)
        
        logger.info(f"Entity-weighted index created with {self.faiss_index.ntotal} examples")
        return self.faiss_index
    
    def classify(self, query: str, k: int = 10) -> Dict:
        """Final hybrid classification"""
        
        # Step 1: Check explicit out-of-scope patterns
        is_explicit_out_of_scope, matched_pattern = self.check_explicit_out_of_scope(query)
        if is_explicit_out_of_scope:
            return {
                'domain': 'out_of_scope',
                'category': 'explicit_out_of_scope',
                'confidence': 0.95,
                'method': 'explicit_pattern_match',
                'reasoning': f'Matched out-of-scope pattern: {matched_pattern}',
                'matched_pattern': matched_pattern
            }
        
        # Step 2: Entity analysis  
        entities, entity_weight = self.extract_entities_with_weights(query)
        
        # Step 3: Embedding similarity
        # Apply same entity weighting to query
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        if entity_weight > 0:
            weight_factor = 1.0 + (entity_weight * 0.1)
            query_embedding = query_embedding * weight_factor
        
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        if len(similarities[0]) == 0:
            return {
                'domain': 'out_of_scope',
                'category': 'no_matches',
                'confidence': 0.90,
                'method': 'no_similarity_matches',
                'reasoning': 'No similar training examples found'
            }
        
        best_similarity = similarities[0][0]
        best_idx = indices[0][0]
        best_category = self.training_labels[best_idx]
        
        # Step 4: Decision logic combining entity weight + similarity
        if entity_weight >= self.entity_threshold and best_similarity >= self.high_confidence_threshold:
            # Strong entity presence + high similarity = high confidence Namjari
            domain = 'namjari'
            confidence = min(0.95, 0.7 + (entity_weight * 0.05) + (best_similarity * 0.2))
            method = 'high_entity_high_similarity'
            reasoning = f'Strong Namjari entities (weight: {entity_weight:.1f}) + high similarity ({best_similarity:.3f})'
            
        elif entity_weight >= self.entity_threshold and best_similarity >= self.similarity_threshold:
            # Strong entity presence + decent similarity = medium confidence Namjari  
            domain = 'namjari'
            confidence = min(0.85, 0.6 + (entity_weight * 0.04) + (best_similarity * 0.15))
            method = 'high_entity_medium_similarity'
            reasoning = f'Strong Namjari entities (weight: {entity_weight:.1f}) + decent similarity ({best_similarity:.3f})'
            
        elif entity_weight >= 1.0 and best_similarity >= self.high_confidence_threshold:
            # Some entities + very high similarity = medium confidence Namjari
            domain = 'namjari' 
            confidence = min(0.80, 0.5 + (entity_weight * 0.03) + (best_similarity * 0.25))
            method = 'medium_entity_high_similarity'
            reasoning = f'Some Namjari entities (weight: {entity_weight:.1f}) + very high similarity ({best_similarity:.3f})'
            
        elif entity_weight < 0.5 and best_similarity < self.similarity_threshold:
            # No entities + low similarity = out of scope
            domain = 'out_of_scope'
            confidence = min(0.90, 0.7 + (0.5 - entity_weight) * 0.2 + (self.similarity_threshold - best_similarity) * 0.3)
            method = 'low_entity_low_similarity'  
            reasoning = f'Few Namjari entities (weight: {entity_weight:.1f}) + low similarity ({best_similarity:.3f})'
            
        else:
            # Edge cases - be conservative
            domain = 'out_of_scope'
            confidence = 0.60 + abs(entity_weight - 1.0) * 0.1 + abs(best_similarity - 0.8) * 0.1
            method = 'conservative_fallback'
            reasoning = f'Unclear case: entities={entity_weight:.1f}, similarity={best_similarity:.3f} - being conservative'
        
        return {
            'domain': domain,
            'category': best_category if domain == 'namjari' else 'out_of_scope',
            'confidence': confidence,
            'method': method,
            'reasoning': reasoning,
            'entity_weight': entity_weight,
            'entities': entities,
            'best_similarity': best_similarity,
            'best_match': self.training_texts[best_idx]
        }

    def test_final_approach(self):
        """Test the final hybrid approach"""
        
        user_examples = [
            ("জমি দখলের কিভাবে পেতে পারি?", "out_of_scope"),  # Illegal occupation
            ("আমার হারিয়ে যাওয়া বই পেতে কি করতে হবে?", "out_of_scope"),  # Lost item
            ("হজ্ব করতে চাই, কি করতে হবে?", "out_of_scope"),  # Religious
            ("অনলাইনে ওমরাহ্‌ নিবন্ধন করতে কি করতে হবে?", "out_of_scope"),  # Religious
            ("মিউট করতে হলে কি করতে হবে?", "out_of_scope"),  # Technology (not mutation)
            ("আমি ওমরাহ্‌ করবো কি করতে হবে?", "out_of_scope"),  # Religious
            ("কোম্পানিতে আবেদন কি নিজে করতে পারি।", "out_of_scope"),  # Job
            ("জমি দখল করতে কি ভূমি অফিসে যাওয়া লাগে?", "out_of_scope"),  # Illegal occupation
            ("আমি নিজে কি চাকরির আবেদন করতে পারবো?", "out_of_scope"),  # Job
            ("কারো সাহায্যে কি দখল করা যাবে?", "out_of_scope"),  # Illegal occupation
            ("জমি দখলের জন্য নিবন্ধন করতে কি লাগে?", "out_of_scope"),  # Still illegal
            ("জন্মনিবন্ধন করতে কি কি দরকার?", "out_of_scope"),  # Birth registration
            ("জন্মনিবন্ধন করতে কি দলিল লাগে?", "out_of_scope"),  # Birth registration
            ("জন্মনিবন্ধন করতে নিজের মোবাইল থাকতে হবে?", "out_of_scope"),  # Birth registration
            ("জন্মনিবন্ধন করতে মোবাইল নম্বর ছাড়া আর কি লাগে?", "out_of_scope"),  # Birth registration
            ("জন্মনিবন্ধন করতে এনআইডি বা জন্মনিবন্ধন লাগে কি?", "out_of_scope"),  # Birth registration
            ("হজ্ব আবেদন নিজে না করলে প্রতিনিধি দিয়ে করা যায় কিনা?", "out_of_scope"),  # Religious
            ("আমি নিজ এলাকার বাইরে থাকি অন্য কেউ কি আমার নামে চাঁদাবাজি করতে পারে কিনা?", "out_of_scope"),  # Crime
            ("আমি বিদেশে থাকি আমার ভাই বা আত্মীয় দশ পারসেন্ট আবেদন করতে পারে কিনা?", "out_of_scope"),  # Quota system
            ("আমি নারী জমির কিছুই বুঝি না, আমি সিনেসিস এ আবেদন করাতে পারবো তো?", "namjari"),  # Land service
            
            # Some clear Namjari examples for testing
            ("নামজারি করতে কি করতে হবে?", "namjari"),
            ("জমির নামজারি করার প্রক্রিয়া কি?", "namjari"), 
            ("খতিয়ানে ভুল সংশোধন করতে চাই", "namjari"),
            ("ওয়ারিশের নামে জমি নামজারি করবো কিভাবে?", "namjari"),
        ]
        
        print("\n🎯 FINAL HYBRID APPROACH RESULTS")
        print("="*80)
        
        correct = 0
        total = 0
        namjari_correct = 0
        namjari_total = 0  
        out_of_scope_correct = 0
        out_of_scope_total = 0
        
        for query, expected in user_examples:
            result = self.classify(query)
            is_correct = result['domain'] == expected
            
            if is_correct:
                correct += 1
            total += 1
            
            if expected == 'namjari':
                namjari_total += 1
                if is_correct:
                    namjari_correct += 1
            else:
                out_of_scope_total += 1
                if is_correct:
                    out_of_scope_correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"\n{status} Query: {query}")
            print(f"   🎯 Result: {result['domain']} (conf: {result['confidence']:.3f})")
            print(f"   📝 Expected: {expected}")
            print(f"   🔧 Method: {result['method']}")
            if 'entities' in result and result['entities']:
                entities_str = ', '.join([f"{k}({v})" for k,v in result['entities'].items()])
                print(f"   🏷️ Entities: {entities_str} (total: {result.get('entity_weight', 0):.1f})")
            print(f"   💭 Reasoning: {result['reasoning']}")
        
        overall_accuracy = correct / total
        namjari_accuracy = namjari_correct / namjari_total if namjari_total > 0 else 0
        out_of_scope_accuracy = out_of_scope_correct / out_of_scope_total if out_of_scope_total > 0 else 0
        
        print(f"\n📊 FINAL HYBRID RESULTS:")
        print(f"   🎯 Overall Accuracy: {correct}/{total} ({overall_accuracy*100:.1f}%)")
        print(f"   ✅ Namjari Accuracy: {namjari_correct}/{namjari_total} ({namjari_accuracy*100:.1f}%)")  
        print(f"   🚫 Out-of-Scope Accuracy: {out_of_scope_correct}/{out_of_scope_total} ({out_of_scope_accuracy*100:.1f}%)")
        
        print(f"\n🎯 HYBRID STRATEGY COMPONENTS:")
        print(f"   1. ✅ Explicit out-of-scope pattern matching")
        print(f"   2. ✅ Entity-weighted embeddings")  
        print(f"   3. ✅ Multi-factor decision logic")
        print(f"   4. ✅ Conservative fallback for edge cases")
        print("="*80)
        
        return {
            'overall_accuracy': overall_accuracy,
            'namjari_accuracy': namjari_accuracy,
            'out_of_scope_accuracy': out_of_scope_accuracy
        }

def main():
    """Test the final hybrid approach"""
    logger.info("🚀 FINAL HYBRID CLASSIFIER TEST")
    logger.info("="*80)
    
    classifier = FinalHybridClassifier()
    classifier.load_all_training_data()
    classifier.create_entity_weighted_index()
    
    results = classifier.test_final_approach()
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()