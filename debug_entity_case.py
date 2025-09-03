#!/usr/bin/env python3
"""Debug the failing entity case"""

import sys
sys.path.append('.')

from entity_weighted_embeddings import EntityWeightedEmbeddingSystem

def debug_failing_case():
    print("=== DEBUGGING FAILING CASE ===")
    
    # Initialize system
    system = EntityWeightedEmbeddingSystem()
    system.load_dataset()
    system.create_embeddings()
    system.build_faiss_index()
    
    # Debug the failing query
    query = "জমি নামজারি করতে কি দলিল লাগে?"
    
    print(f"Query: {query}")
    
    # Check entity extraction
    entities = system.entity_extractor.extract_entities(query)
    entity_score = system.entity_extractor.calculate_entity_score(query)
    
    print(f"Found entities: {entities}")
    print(f"Entity score: {entity_score}")
    
    # Check embedding similarity results
    results = system.search_similar(query, k=10)
    
    print(f"\nTop 10 similarity results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['similarity_score']:.3f} | "
              f"Category: {result['category']} | "
              f"Text: '{result['text'][:50]}...'")
    
    # Check classification result
    classification = system.classify_query(query)
    print(f"\nClassification result:")
    for key, value in classification.items():
        print(f"  {key}: {value}")
    
    # Check threshold adjustments
    print(f"\nThreshold analysis:")
    print(f"  Entity score {entity_score:.2f} > 5.0? {entity_score > 5.0}")
    print(f"  Entity score {entity_score:.2f} > 2.0? {entity_score > 2.0}")
    
    # Find the best namjari match manually
    best_namjari = None
    for result in results:
        if result['category'] != 'out_of_scope':
            best_namjari = result
            break
    
    if best_namjari:
        print(f"  Best namjari match: {best_namjari['category']} (score: {best_namjari['similarity_score']:.3f})")
    else:
        print(f"  No namjari matches found!")

if __name__ == "__main__":
    debug_failing_case()