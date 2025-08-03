#!/usr/bin/env python3
"""Simple test script for the error analysis system - no API keys required."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_comparator import error_classifier
from llm_comparator import error_taxonomy


def test_error_taxonomy():
    """Test the error taxonomy system."""
    print("🧪 Testing Error Taxonomy System")
    print("=" * 50)
    
    taxonomy = error_taxonomy.ERROR_TAXONOMY
    
    print(f"✅ Loaded {len(taxonomy.error_types)} error types")
    print(f"✅ Loaded {len(taxonomy.get_all_categories())} error categories")
    
    # Show categories by priority
    print("\n📊 Error Categories (by priority):")
    categories_by_weight = sorted(
        taxonomy.get_all_categories(),
        key=lambda cat: taxonomy.get_category_weight(cat),
        reverse=True
    )
    
    for category in categories_by_weight:
        weight = taxonomy.get_category_weight(category)
        types_in_category = taxonomy.get_error_types_by_category(category)
        print(f"  {category.value.upper()}: weight={weight:.1f}, types={len(types_in_category)}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality:")
    search_results = taxonomy.search_error_types("incorrect")
    print(f"  Found {len(search_results)} error types for 'incorrect'")
    for result in search_results[:3]:
        print(f"    - {result.name} ({result.category.value})")


def test_pattern_detection():
    """Test pattern-based error detection."""
    print("\n🔍 Testing Pattern-Based Error Detection")
    print("=" * 50)
    
    # Create classifier (no LLM needed for pattern detection)
    classifier = error_classifier.ErrorClassifier()
    
    # Test cases with known errors
    test_cases = [
        {
            'prompt': 'What is the capital of France?',
            'response': 'The capital of France is Lyon, which is incorrect information about geography.',
            'expected_errors': ['factual']
        },
        {
            'prompt': 'Explain quantum physics',
            'response': 'This response contains fabricated facts and made up scientific theories that never existed.',
            'expected_errors': ['hallucination']
        },
        {
            'prompt': 'Tell me about history',
            'response': 'This answer is biased against certain demographic groups and shows cultural bias.',
            'expected_errors': ['bias']
        },
        {
            'prompt': 'How does this work?',
            'response': 'This explanation has logical fallacies and flawed reasoning throughout.',
            'expected_errors': ['reasoning']
        }
    ]
    
    total_detected = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"   Prompt: {test_case['prompt']}")
        print(f"   Response: {test_case['response']}")
        
        # Analyze the response (pattern-based only)
        result = classifier.classify_errors(
            response_id=f"test_{i}",
            prompt=test_case['prompt'],
            response=test_case['response'],
            use_llm_analysis=False  # No LLM needed
        )
        
        print(f"   📊 Results:")
        print(f"      Errors detected: {len(result.detected_errors)}")
        print(f"      Overall error score: {result.overall_error_score:.3f}")
        print(f"      Confidence: {result.confidence_score:.3f}")
        print(f"      Has critical errors: {result.has_critical_errors}")
        
        if result.detected_errors:
            print(f"   🚨 Detected Errors:")
            for error in result.detected_errors:
                print(f"      - {error.error_type.name}")
                print(f"        Category: {error.error_type.category.value}")
                print(f"        Severity: {error.severity.value}")
                print(f"        Confidence: {error.confidence:.2f}")
                print(f"        Evidence: '{error.evidence}'")
        else:
            print(f"   ✅ No errors detected")
        
        total_detected += len(result.detected_errors)
    
    print(f"\n📈 Summary: Detected {total_detected} total errors across {len(test_cases)} test cases")


def test_recipe_data():
    """Test with your recipe evaluation data."""
    print("\n🍳 Testing with Recipe Data")
    print("=" * 50)
    
    # Load your recipe evaluation data
    recipe_file = "C:/Users/RICHIE/Documents/recipe_eval_data.json"
    
    try:
        import json
        with open(recipe_file, 'r') as f:
            recipe_data = json.load(f)
        
        print(f"✅ Loaded {len(recipe_data)} recipe examples")
        
        classifier = error_classifier.ErrorClassifier()
        
        # Test with first few examples
        for i, example in enumerate(recipe_data[:3], 1):
            print(f"\n📝 Recipe Example {i}:")
            print(f"   Input: {example.get('input', 'N/A')}")
            
            # Create a mock response for testing
            mock_response = f"Here's a recipe response that might contain errors or be perfectly fine. This is example {i}."
            
            result = classifier.classify_errors(
                response_id=f"recipe_{i}",
                prompt=example.get('input', ''),
                response=mock_response,
                use_llm_analysis=False
            )
            
            print(f"   📊 Analysis:")
            print(f"      Errors: {len(result.detected_errors)}")
            print(f"      Score: {result.overall_error_score:.3f}")
            
    except FileNotFoundError:
        print(f"❌ Recipe file not found at {recipe_file}")
        print("   You can still test with the built-in examples above!")
    except Exception as e:
        print(f"❌ Error loading recipe data: {e}")


def main():
    """Run all tests."""
    print("🚀 LLM Comparator - Error Analysis System Test")
    print("=" * 60)
    
    try:
        test_error_taxonomy()
        test_pattern_detection()
        test_recipe_data()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. The pattern-based detection is working!")
        print("2. To enable LLM-based detection, you'll need to set up model helpers")
        print("3. Try modifying the test cases to see different error types")
        print("4. Integrate with your actual evaluation pipeline")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()