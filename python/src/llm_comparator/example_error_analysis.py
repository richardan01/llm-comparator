#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example usage of the enhanced LLM Comparator with systematic error analysis."""

import json
from typing import List

from llm_comparator import enhanced_comparison
from llm_comparator import error_classifier
from llm_comparator import error_taxonomy
from llm_comparator import model_helper
from llm_comparator import llm_judge_runner
from llm_comparator import rationale_bullet_generator
from llm_comparator import rationale_cluster_generator
from llm_comparator import types


def create_sample_data() -> List[types.LLMJudgeInput]:
    """Create sample evaluation data for demonstration."""
    return [
        {
            'prompt': 'What is the capital of France?',
            'response_a': 'The capital of France is Paris, which is located in the north-central part of the country.',
            'response_b': 'The capital of France is Lyon. It is a beautiful city with many museums and cultural attractions.',
        },
        {
            'prompt': 'Explain how photosynthesis works.',
            'response_a': 'Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This occurs primarily in the chloroplasts of plant cells.',
            'response_b': 'Photosynthesis is when plants eat sunlight and turn it into food. The process happens because plants are hungry and need energy from the sun to survive.',
        },
        {
            'prompt': 'What are the health benefits of regular exercise?',
            'response_a': 'Regular exercise provides numerous health benefits including improved cardiovascular health, stronger muscles and bones, better mental health, and reduced risk of chronic diseases.',
            'response_b': 'Exercise is good for you and makes you feel better. It helps with weight loss and can prevent some diseases, though the exact mechanisms are not well understood.',
        }
    ]


def demonstrate_error_taxonomy():
    """Demonstrate the error taxonomy system."""
    print("=== Error Taxonomy Demonstration ===")
    
    taxonomy = error_taxonomy.ERROR_TAXONOMY
    
    print(f"Total error categories: {len(taxonomy.get_all_categories())}")
    print(f"Total error types: {len(taxonomy.error_types)}")
    
    print("\nError Categories (by priority):")
    categories_by_weight = sorted(
        taxonomy.get_all_categories(),
        key=lambda cat: taxonomy.get_category_weight(cat),
        reverse=True
    )
    
    for category in categories_by_weight:
        weight = taxonomy.get_category_weight(category)
        types_in_category = taxonomy.get_error_types_by_category(category)
        print(f"  {category.value.upper()}: {weight:.1f} ({len(types_in_category)} types)")
    
    print("\nExample error types:")
    for error_type in list(taxonomy.error_types.values())[:5]:
        print(f"  - {error_type.name} ({error_type.category.value}): {error_type.severity_default.value}")


def demonstrate_pattern_based_detection():
    """Demonstrate pattern-based error detection."""
    print("\n=== Pattern-Based Error Detection ===")
    
    classifier = error_classifier.ErrorClassifier()
    
    test_responses = [
        "The capital of France is Lyon, which is incorrect information.",
        "This response contains fabricated facts about historical events.",
        "The answer is biased against certain demographic groups.",
        "This explanation has logical fallacies and flawed reasoning.",
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\nAnalyzing response {i+1}: '{response}'")
        
        result = classifier.classify_errors(
            response_id=f"test_{i}",
            prompt="Test prompt",
            response=response,
            use_llm_analysis=False  # Only pattern-based for this demo
        )
        
        print(f"  Detected {len(result.detected_errors)} errors")
        print(f"  Overall error score: {result.overall_error_score:.3f}")
        print(f"  Confidence: {result.confidence_score:.3f}")
        
        for error in result.detected_errors:
            print(f"    - {error.error_type.name}: {error.confidence:.2f} confidence")


def demonstrate_enhanced_comparison():
    """Demonstrate the enhanced comparison with error analysis."""
    print("\n=== Enhanced Comparison with Error Analysis ===")
    
    # Create sample data
    inputs = create_sample_data()
    
    # Note: In a real scenario, you would initialize these with actual model helpers
    print("Note: This is a demonstration of the API structure.")
    print("In practice, you would initialize with actual model helpers:")
    print("""
    # Example initialization (requires actual API keys/models):
    generation_model = model_helper.GenerationModelHelper(...)
    embedding_model = model_helper.EmbeddingModelHelper(...)
    
    judge = llm_judge_runner.LLMJudgeRunner(generation_model)
    bulletizer = rationale_bullet_generator.RationaleBulletGenerator(generation_model)
    clusterer = rationale_cluster_generator.RationaleClusterGenerator(generation_model, embedding_model)
    error_classifier_instance = enhanced_comparison.create_error_classifier_with_judge(judge)
    
    # Run enhanced comparison
    results = enhanced_comparison.run_with_error_analysis(
        inputs=inputs,
        judge=judge,
        bulletizer=bulletizer,
        clusterer=clusterer,
        error_classifier_instance=error_classifier_instance,
        enable_error_analysis=True,
        use_llm_error_detection=True
    )
    
    # Save results
    enhanced_comparison.write_enhanced(results, 'enhanced_results.json')
    """)
    
    print(f"Sample data contains {len(inputs)} examples:")
    for i, input_data in enumerate(inputs):
        print(f"  {i+1}. {input_data['prompt']}")


def demonstrate_error_analysis_features():
    """Demonstrate key features of the error analysis system."""
    print("\n=== Error Analysis Features ===")
    
    print("✓ Comprehensive Error Taxonomy:")
    print("  - 10 error categories (Factual, Reasoning, Coherence, etc.)")
    print("  - 16+ specific error types")
    print("  - 5 severity levels (Critical, High, Medium, Low, Negligible)")
    
    print("\n✓ Automated Error Detection:")
    print("  - Pattern-based detection using regex")
    print("  - LLM-based detection for complex errors")
    print("  - Confidence scoring for each detection")
    
    print("\n✓ Systematic Analysis:")
    print("  - Overall error scores with category weighting")
    print("  - Per-category error analysis")
    print("  - Critical error flagging")
    print("  - Comparative analysis between models")
    
    print("\n✓ Enhanced Output Format:")
    print("  - Backward compatible with existing LLM Comparator")
    print("  - Rich error analysis metadata")
    print("  - Detailed error summaries and statistics")


if __name__ == "__main__":
    print("LLM Comparator - Enhanced Error Analysis System")
    print("=" * 50)
    
    demonstrate_error_taxonomy()
    demonstrate_pattern_based_detection()
    demonstrate_enhanced_comparison()
    demonstrate_error_analysis_features()
    
    print("\n" + "=" * 50)
    print("Phase 1 Implementation Complete!")
    print("\nNext steps:")
    print("1. Set up actual model helpers (API keys, endpoints)")
    print("2. Test with real evaluation data")
    print("3. Customize error taxonomy for your specific use case")
    print("4. Integrate with your existing evaluation pipeline")
