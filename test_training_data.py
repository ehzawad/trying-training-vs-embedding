#!/usr/bin/env python3
"""
Test Final Model Against Training Data
Check if model can distinguish categories from its own training data
Critical test: Does it miss seen data or confuse categories?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_against_training_data():
    """
    Test model against ALL data it was trained on
    Critical test: Can it handle its own training categories?
    """
    
    # Load the trained model
    model_path = "models/namjari-final/final"
    
    try:
        logger.info(f"Loading trained model: {model_path}")
        model = SentenceTransformer(model_path)
    except Exception as e:
        logger.error(f"Model not found: {e}")
        logger.info("Please run train_final.py first")
        return None
    
    # Load ALL training data from namjari_questions folder
    data_dir = Path("namjari_questions")
    categories_data = {}
    all_questions = []
    all_categories = []
    
    logger.info("Loading ALL training data...")
    
    for csv_file in data_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        category = csv_file.stem.replace('namjari_', '')
        questions = df['question'].tolist()
        
        categories_data[category] = questions
        all_questions.extend(questions)
        all_categories.extend([category] * len(questions))
    
    logger.info(f"Loaded {len(all_questions)} questions from {len(categories_data)} categories")
    
    # Test 1: Can model distinguish between different categories?
    logger.info("\n=== TEST 1: CATEGORY DISTINCTION (TRAINING DATA) ===")
    
    # Sample 5 questions from each category for testing
    category_samples = {}
    for category, questions in categories_data.items():
        category_samples[category] = questions[:5]  # First 5 from each
    
    # Calculate inter-category similarities
    category_similarities = {}
    
    for cat1, questions1 in category_samples.items():
        for cat2, questions2 in category_samples.items():
            if cat1 >= cat2:  # Avoid duplicates
                continue
                
            emb1 = model.encode(questions1)
            emb2 = model.encode(questions2)
            similarity = cosine_similarity(emb1, emb2).mean()
            
            category_similarities[(cat1, cat2)] = similarity
    
    # Show results sorted by similarity
    sorted_similarities = sorted(category_similarities.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("Inter-category similarities (higher = more confused):")
    for (cat1, cat2), sim in sorted_similarities:
        status = "🚨" if sim > 0.8 else "⚠️" if sim > 0.6 else "✅"
        logger.info(f"  {status} {sim:.3f}: {cat1} <-> {cat2}")
    
    # Test 2: Intra-category consistency  
    logger.info("\n=== TEST 2: INTRA-CATEGORY CONSISTENCY ===")
    
    intra_category_scores = {}
    
    for category, questions in category_samples.items():
        if len(questions) < 2:
            continue
            
        embeddings = model.encode(questions)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Remove diagonal and get average within-category similarity
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        avg_similarity = upper_triangle.mean()
        
        intra_category_scores[category] = avg_similarity
        
        status = "✅" if avg_similarity > 0.7 else "⚠️" if avg_similarity > 0.5 else "❌"
        logger.info(f"  {status} {avg_similarity:.3f}: {category} (within-category)")
    
    # Test 3: Specific Category Pairs (Most Confusing)
    logger.info("\n=== TEST 3: MOST CONFUSING CATEGORY PAIRS ===")
    
    most_confusing = sorted_similarities[:5]  # Top 5 most similar
    
    for (cat1, cat2), sim in most_confusing:
        logger.info(f"\n🔍 Testing: {cat1} vs {cat2} (similarity: {sim:.3f})")
        
        # Show specific example confusions
        sample1 = categories_data[cat1][:3]
        sample2 = categories_data[cat2][:3]
        
        for i, q1 in enumerate(sample1):
            for j, q2 in enumerate(sample2):
                emb1 = model.encode([q1])
                emb2 = model.encode([q2])
                pair_sim = cosine_similarity(emb1, emb2)[0][0]
                
                if pair_sim > 0.7:  # High confusion
                    logger.info(f"  🚨 {pair_sim:.3f}: '{q1[:40]}...' vs '{q2[:40]}...'")
    
    # Test 4: Random Sampling Test
    logger.info("\n=== TEST 4: RANDOM SAMPLING TEST ===")
    
    # Randomly sample questions and see if model can identify their categories
    n_samples = 20
    random_indices = np.random.choice(len(all_questions), n_samples, replace=False)
    
    correct_predictions = 0
    
    for idx in random_indices:
        test_question = all_questions[idx]
        true_category = all_categories[idx]
        
        # Find most similar category by comparing to category representatives
        best_similarity = -1
        predicted_category = None
        
        for category, questions in category_samples.items():
            rep_embeddings = model.encode(questions)
            test_embedding = model.encode([test_question])
            
            similarity = cosine_similarity(test_embedding, rep_embeddings).max()
            
            if similarity > best_similarity:
                best_similarity = similarity
                predicted_category = category
        
        is_correct = predicted_category == true_category
        if is_correct:
            correct_predictions += 1
            
        status = "✅" if is_correct else "❌"
        logger.info(f"  {status} '{test_question[:40]}...' -> {predicted_category} (true: {true_category}, sim: {best_similarity:.3f})")
    
    accuracy = correct_predictions / n_samples
    logger.info(f"\nCategory prediction accuracy: {correct_predictions}/{n_samples} ({accuracy*100:.1f}%)")
    
    # Summary Assessment
    logger.info("\n=== TRAINING DATA PERFORMANCE SUMMARY ===")
    
    avg_inter_category = np.mean([sim for _, sim in category_similarities.items()])
    avg_intra_category = np.mean(list(intra_category_scores.values()))
    separation_score = avg_intra_category - avg_inter_category
    
    logger.info(f"Average inter-category similarity: {avg_inter_category:.3f}")
    logger.info(f"Average intra-category similarity: {avg_intra_category:.3f}")
    logger.info(f"Category separation score: {separation_score:.3f}")
    logger.info(f"Category prediction accuracy: {accuracy*100:.1f}%")
    
    # Overall assessment
    if accuracy > 0.8 and separation_score > 0.2:
        logger.info("🏆 EXCELLENT: Model learned training data well")
        grade = "A"
    elif accuracy > 0.6 and separation_score > 0.1:
        logger.info("✅ GOOD: Model shows reasonable learning")
        grade = "B"
    elif accuracy > 0.4:
        logger.info("⚠️ FAIR: Model has some understanding")
        grade = "C"
    else:
        logger.info("❌ POOR: Model failed to learn training data")
        grade = "F"
    
    return {
        'accuracy': accuracy,
        'separation_score': separation_score,
        'avg_inter_category': avg_inter_category,
        'avg_intra_category': avg_intra_category,
        'grade': grade,
        'most_confusing_pairs': most_confusing[:3]
    }

def test_specific_examples():
    """
    Test model on specific examples to understand its behavior
    """
    logger.info("\n=== SPECIFIC EXAMPLE TESTS ===")
    
    model = SentenceTransformer("models/namjari-final/final")
    
    # Test examples from different categories
    test_cases = [
        ("নামজারি করতে কি করতে হবে?", "application_procedure"),
        ("নামজারির ফি কত টাকা?", "fee"),
        ("নামজারি করতে কি দলিল লাগে?", "required_documents"),
        ("নামজারি কখন করতে হয়?", "eligibility"),
        ("নামজারির স্ট্যাটাস চেক করতে চাই", "status_check"),
    ]
    
    # Load category representatives
    data_dir = Path("namjari_questions")
    category_reps = {}
    
    for csv_file in data_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        category = csv_file.stem.replace('namjari_', '')
        questions = df['question'].head(3).tolist()  # 3 representatives
        category_reps[category] = questions
    
    logger.info("Testing specific examples against category representatives:")
    
    for test_question, expected_category in test_cases:
        logger.info(f"\n🔍 Testing: '{test_question}'")
        logger.info(f"   Expected category: {expected_category}")
        
        test_emb = model.encode([test_question])
        
        # Find best matching category
        best_sim = -1
        best_category = None
        
        similarities = {}
        for category, rep_questions in category_reps.items():
            rep_emb = model.encode(rep_questions)
            sim = cosine_similarity(test_emb, rep_emb).max()
            similarities[category] = sim
            
            if sim > best_sim:
                best_sim = sim
                best_category = category
        
        # Show top 3 matches
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        is_correct = best_category == expected_category
        status = "✅" if is_correct else "❌"
        
        logger.info(f"   {status} Predicted: {best_category} (similarity: {best_sim:.3f})")
        logger.info("   Top 3 matches:")
        for i, (cat, sim) in enumerate(sorted_matches):
            marker = "→" if cat == expected_category else " "
            logger.info(f"     {i+1}. {marker} {cat}: {sim:.3f}")

if __name__ == "__main__":
    try:
        # Test against all training data
        results = test_against_training_data()
        
        if results:
            # Test specific examples
            test_specific_examples()
            
            print("\n" + "="*80)
            print("📊 TRAINING DATA PERFORMANCE TEST")
            print("="*80)
            print(f"📈 Category Prediction Accuracy: {results['accuracy']*100:.1f}%")
            print(f"📈 Category Separation Score: {results['separation_score']:.3f}")
            print(f"📈 Inter-category Similarity: {results['avg_inter_category']:.3f}")
            print(f"📈 Intra-category Similarity: {results['avg_intra_category']:.3f}")
            print(f"📈 Overall Grade: {results['grade']}")
            
            print("\n🔍 Most Confusing Category Pairs:")
            for (cat1, cat2), sim in results['most_confusing_pairs']:
                print(f"   {sim:.3f}: {cat1} <-> {cat2}")
            
            print("\n🎯 Key Insights:")
            if results['accuracy'] > 0.7:
                print("   ✅ Model learned training data categories well")
            else:
                print("   ⚠️ Model struggles even with training data categories")
                
            if results['separation_score'] > 0.2:
                print("   ✅ Good separation between categories")
            else:
                print("   ⚠️ Poor separation - categories too similar")
            
            print("="*80)
            
    except Exception as e:
        logger.error(f"Training data test failed: {e}")
        raise
