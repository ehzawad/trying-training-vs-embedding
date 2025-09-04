#!/usr/bin/env python3
"""
Bengali Entity-First Hybrid Legal Text Classification
Addresses fundamental limitations of pure embedding approaches
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BengaliLegalNER:
    """
    Custom Bengali Legal Domain Named Entity Recognition
    Focuses on domain-specific legal entities
    """
    
    def __init__(self):
        # Domain entity patterns with weights
        self.entity_patterns = {
            # Core legal document entities (high weight)
            'legal_documents': {
                'patterns': [
                    r'দলিল', r'সনদ', r'কাগজপত্র', r'ডকুমেন্ট',
                    r'তফসিল', r'পরচা', r'খতিয়ান', r'রেকর্ড'
                ],
                'weight': 10.0,
                'category_hints': ['required_documents', 'inheritance_documents', 'hearing_documents']
            },
            
            # Land/Property entities (high weight)
            'property': {
                'patterns': [
                    r'নামজারি', r'মিউটেশন', r'ভূমি', r'জমি', r'সম্পত্তি',
                    r'মৌজা', r'দাগ', r'খতিয়ান', r'জরিপ'
                ],
                'weight': 10.0,
                'category_hints': ['application_procedure', 'process', 'khatian_copy', 'khatian_correction']
            },
            
            # Fee and cost entities (medium weight)
            'financial': {
                'patterns': [
                    r'ফি', r'খরচ', r'টাকা', r'পেমেন্ট', r'শুল্ক',
                    r'ভ্যাট', r'কর', r'রসিদ'
                ],
                'weight': 8.0,
                'category_hints': ['fee']
            },
            
            # Process and procedure entities (medium weight)
            'procedure': {
                'patterns': [
                    r'আবেদন', r'প্রক্রিয়া', r'পদ্ধতি', r'নিয়ম',
                    r'শর্ত', r'যোগ্যতা', r'এলিজিবিলিটি'
                ],
                'weight': 6.0,
                'category_hints': ['application_procedure', 'eligibility', 'process']
            },
            
            # Status and tracking entities (medium weight)
            'status_tracking': {
                'patterns': [
                    r'স্ট্যাটাস', r'অবস্থা', r'চেক', r'ট্র্যাক',
                    r'খোঁজ', r'তথ্য', r'জানতে'
                ],
                'weight': 7.0,
                'category_hints': ['status_check']
            },
            
            # Legal proceedings entities (medium weight)
            'legal_proceedings': {
                'patterns': [
                    r'শুনানি', r'নোটিশ', r'আপিল', r'প্রতিনিধি',
                    r'আইনজীবী', r'উকিল', r'কোর্ট', r'আদালত'
                ],
                'weight': 7.0,
                'category_hints': ['hearing_notification', 'hearing_documents', 'by_representative', 'rejected_appeal']
            },
            
            # Inheritance entities (medium weight)
            'inheritance': {
                'patterns': [
                    r'ওয়ারিশ', r'উত্তরাধিকার', r'মৃত', r'মরহুম',
                    r'স্বামী', r'স্ত্রী', r'পুত্র', r'কন্যা'
                ],
                'weight': 8.0,
                'category_hints': ['inheritance_documents']
            },
            
            # Registration entities (medium weight)
            'registration': {
                'patterns': [
                    r'নিবন্ধন', r'রেজিস্ট্রেশন', r'সাব-রেজিস্ট্রার',
                    r'রেজিস্ট্রি', r'অফিস'
                ],
                'weight': 6.0,
                'category_hints': ['registration']
            },
            
            # Negative entities (strong negative weight)
            'out_of_scope': {
                'patterns': [
                    r'হজ্?ব', r'ওমরাহ', r'জন্মনিবন্ধন', r'চাকরি', r'চাকুরি',
                    r'পাসপোর্ট', r'ভিসা', r'মোবাইল', r'ইন্টারনেট',
                    r'আবহাওয়া', r'খেলা', r'সিনেমা', r'দখল', r'চাঁদাবাজি'
                ],
                'weight': -15.0,
                'category_hints': ['out_of_scope']
            }
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, data in self.entity_patterns.items():
            self.compiled_patterns[entity_type] = {
                'patterns': [re.compile(pattern) for pattern in data['patterns']],
                'weight': data['weight'],
                'category_hints': data['category_hints']
            }
    
    def extract_entities(self, text: str) -> Dict[str, float]:
        """Extract entities and return weighted scores"""
        entity_scores = {}
        total_score = 0.0
        
        for entity_type, data in self.compiled_patterns.items():
            type_score = 0.0
            matched_patterns = []
            
            for pattern in data['patterns']:
                matches = pattern.findall(text)
                if matches:
                    # Count frequency and apply weight
                    frequency = len(matches)
                    pattern_score = frequency * data['weight']
                    type_score += pattern_score
                    matched_patterns.extend(matches)
            
            if type_score != 0:
                entity_scores[entity_type] = {
                    'score': type_score,
                    'matches': matched_patterns,
                    'category_hints': data['category_hints']
                }
                total_score += type_score
        
        return {
            'entities': entity_scores,
            'total_score': total_score,
            'is_out_of_scope': total_score < -10.0,
            'is_strong_positive': total_score > 15.0
        }

class WeightedAttentionEmbedder:
    """
    Custom embedding approach that weights tokens based on domain importance
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.ner = BengaliLegalNER()
        
        # Create domain-specific token weights
        self.token_weights = {}
        for entity_type, data in self.ner.entity_patterns.items():
            weight = abs(data['weight']) / 10.0  # Normalize for attention
            for pattern in data['patterns']:
                # Remove regex special characters for token matching
                token = re.sub(r'[\\^$.*+?{}[\]|()]', '', pattern)
                self.token_weights[token] = weight
    
    def encode_with_attention(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with entity-weighted attention
        """
        # Get base embeddings
        base_embeddings = self.model.encode(texts)
        
        # Apply entity weighting
        weighted_embeddings = []
        
        for i, text in enumerate(texts):
            # Extract entities
            entity_info = self.ner.extract_entities(text)
            
            # Get base embedding
            base_emb = base_embeddings[i]
            
            # Apply entity-based modification
            if entity_info['total_score'] != 0:
                # Create attention vector based on entities found
                attention_boost = min(abs(entity_info['total_score']) / 50.0, 2.0)  # Cap at 2x boost
                
                # Boost embedding magnitude for strong entities
                if entity_info['is_strong_positive']:
                    weighted_emb = base_emb * (1.0 + attention_boost)
                elif entity_info['is_out_of_scope']:
                    # Create distinctive out-of-scope signature
                    weighted_emb = base_emb * 0.1  # Heavily dampen
                else:
                    weighted_emb = base_emb * (1.0 + attention_boost * 0.5)
            else:
                weighted_emb = base_emb
            
            weighted_embeddings.append(weighted_emb)
        
        return np.array(weighted_embeddings)

class EntityFirstHybridClassifier:
    """
    Main hybrid classifier that combines:
    1. Entity extraction and scoring
    2. Weighted attention embeddings  
    3. Traditional ML classification
    4. Rule-based filtering
    """
    
    def __init__(self):
        self.ner = BengaliLegalNER()
        self.embedder = WeightedAttentionEmbedder()
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            analyzer='char_wb',  # Character n-grams work well for Bengali
            min_df=2
        )
        self.ml_classifier = LogisticRegression(
            multi_class='ovr',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        self.category_mapping = {}
        self.is_trained = False
        
    def load_data(self, data_dir: str = "namjari_questions"):
        """Load training data"""
        logger.info("Loading training data...")
        
        data_dir = Path(data_dir)
        texts = []
        labels = []
        
        for csv_file in data_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            category = csv_file.stem.replace('namjari_', '')
            
            questions = df['question'].tolist()
            texts.extend(questions)
            labels.extend([category] * len(questions))
        
        # Add synthetic out-of-scope examples
        out_of_scope_examples = [
            "হজ্ব করতে কি করতে হবে?",
            "জন্মনিবন্ধন করতে কি দলিল লাগে?", 
            "চাকরির আবেদন কিভাবে করবো?",
            "আজকের আবহাওয়া কেমন?",
            "মোবাইল রিচার্জ করতে চাই",
            "জমি দখলের কিভাবে পেতে পারি?",
            "পাসপোর্ট করতে কি লাগে?",
            "ভিসার জন্য কি করতে হবে?"
        ]
        
        texts.extend(out_of_scope_examples)
        labels.extend(['out_of_scope'] * len(out_of_scope_examples))
        
        logger.info(f"Loaded {len(texts)} texts with {len(set(labels))} unique categories")
        
        return texts, labels
    
    def create_features(self, texts: List[str]) -> np.ndarray:
        """Create hybrid feature vectors"""
        logger.info("Creating hybrid features...")
        
        # 1. Entity-based features
        entity_features = []
        for text in texts:
            entity_info = self.ner.extract_entities(text)
            
            # Create feature vector from entities
            feature_vec = []
            
            # Basic entity scores
            for entity_type in self.ner.entity_patterns.keys():
                if entity_type in entity_info['entities']:
                    feature_vec.append(entity_info['entities'][entity_type]['score'])
                else:
                    feature_vec.append(0.0)
            
            # Meta features
            feature_vec.extend([
                entity_info['total_score'],
                1.0 if entity_info['is_out_of_scope'] else 0.0,
                1.0 if entity_info['is_strong_positive'] else 0.0,
                len(entity_info['entities'])  # Number of entity types found
            ])
            
            entity_features.append(feature_vec)
        
        entity_features = np.array(entity_features)
        
        # 2. Weighted attention embeddings
        embeddings = self.embedder.encode_with_attention(texts)
        
        # 3. TF-IDF features
        if self.is_trained:
            tfidf_features = self.tfidf.transform(texts).toarray()
        else:
            tfidf_features = self.tfidf.fit_transform(texts).toarray()
        
        # Combine all features
        combined_features = np.hstack([
            entity_features,
            embeddings,
            tfidf_features
        ])
        
        logger.info(f"Created feature matrix: {combined_features.shape}")
        return combined_features
    
    def train(self, texts: List[str], labels: List[str]):
        """Train the hybrid classifier"""
        logger.info("Training entity-first hybrid classifier...")
        
        # Create features
        features = self.create_features(texts)
        
        # Train ML classifier
        self.ml_classifier.fit(features, labels)
        
        # Create category mapping for interpretability
        unique_categories = sorted(set(labels))
        self.category_mapping = {i: cat for i, cat in enumerate(unique_categories)}
        
        self.is_trained = True
        logger.info(f"Training complete. Categories: {unique_categories}")
    
    def predict_with_reasoning(self, text: str) -> Dict:
        """Predict with detailed reasoning"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # 1. Entity analysis
        entity_info = self.ner.extract_entities(text)
        
        # 2. Quick rule-based filtering
        if entity_info['is_out_of_scope']:
            return {
                'category': 'out_of_scope',
                'confidence': 0.95,
                'method': 'entity_rule_based',
                'reasoning': f"Strong negative entities detected (score: {entity_info['total_score']:.2f})",
                'entities': entity_info['entities']
            }
        
        # 3. Strong positive entity check
        if entity_info['is_strong_positive'] and entity_info['entities']:
            # Find most likely category from entity hints
            category_scores = {}
            for entity_type, data in entity_info['entities'].items():
                for hint in data['category_hints']:
                    category_scores[hint] = category_scores.get(hint, 0) + abs(data['score'])
            
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                if best_category[1] > 20.0:  # High confidence threshold
                    return {
                        'category': best_category[0],
                        'confidence': min(0.95, best_category[1] / 50.0),
                        'method': 'entity_rule_based',
                        'reasoning': f"Strong positive entities for {best_category[0]} (score: {best_category[1]:.2f})",
                        'entities': entity_info['entities']
                    }
        
        # 4. ML classification
        features = self.create_features([text])
        proba = self.ml_classifier.predict_proba(features)[0]
        predicted_idx = np.argmax(proba)
        confidence = proba[predicted_idx]
        predicted_category = self.ml_classifier.classes_[predicted_idx]
        
        return {
            'category': predicted_category,
            'confidence': confidence,
            'method': 'hybrid_ml',
            'reasoning': f"ML classification with {confidence:.3f} confidence",
            'entities': entity_info['entities'],
            'all_probabilities': {
                self.ml_classifier.classes_[i]: proba[i] 
                for i in range(len(proba))
            }
        }
    
    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> Dict:
        """Comprehensive evaluation"""
        logger.info("Evaluating hybrid classifier...")
        
        predictions = []
        detailed_results = []
        
        for text, true_label in zip(test_texts, test_labels):
            result = self.predict_with_reasoning(text)
            predictions.append(result['category'])
            detailed_results.append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'true_label': true_label,
                'predicted': result['category'],
                'confidence': result['confidence'],
                'method': result['method'],
                'correct': result['category'] == true_label
            })
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        
        # Method breakdown
        method_stats = {}
        for result in detailed_results:
            method = result['method']
            if method not in method_stats:
                method_stats[method] = {'total': 0, 'correct': 0}
            method_stats[method]['total'] += 1
            if result['correct']:
                method_stats[method]['correct'] += 1
        
        # Calculate method accuracies
        for method in method_stats:
            stats = method_stats[method]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'overall_accuracy': accuracy,
            'classification_report': report,
            'method_breakdown': method_stats,
            'detailed_results': detailed_results
        }

def main():
    """Main execution"""
    logger.info("🚀 Starting Entity-First Bengali Legal Classification")
    
    # Initialize classifier
    classifier = EntityFirstHybridClassifier()
    
    # Load data
    texts, labels = classifier.load_data()
    
    # Split data (80% train, 20% test)
    split_idx = int(0.8 * len(texts))
    indices = np.random.permutation(len(texts))
    
    train_texts = [texts[i] for i in indices[:split_idx]]
    train_labels = [labels[i] for i in indices[:split_idx]]
    test_texts = [texts[i] for i in indices[split_idx:]]
    test_labels = [labels[i] for i in indices[split_idx:]]
    
    logger.info(f"Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    # Train
    classifier.train(train_texts, train_labels)
    
    # Evaluate
    results = classifier.evaluate(test_texts, test_labels)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"🎯 ENTITY-FIRST HYBRID CLASSIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
    
    print(f"\n📊 Method Breakdown:")
    for method, stats in results['method_breakdown'].items():
        print(f"  {method}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.3f}")
    
    print(f"\n📈 Per-Category Performance:")
    report = results['classification_report']
    for category in sorted(report.keys()):
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = report[category]
            print(f"  {category}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")
    
    print(f"\n🔍 Sample Results:")
    for result in results['detailed_results'][:10]:
        status = "✅" if result['correct'] else "❌"
        print(f"  {status} '{result['text']}' -> {result['predicted']} ({result['confidence']:.3f}, {result['method']})")
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()