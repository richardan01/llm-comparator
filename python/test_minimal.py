#!/usr/bin/env python3
"""Minimal working test for error analysis - tests only what exists."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports to see what's available."""
    print("🔍 Testing Basic Imports")
    print("=" * 40)
    
    modules_to_test = [
        'error_taxonomy',
        'comparison', 
        'types',
        'llm_judge_runner'
    ]
    
    available_modules = []
    
    for module_name in modules_to_test:
        try:
            module = __import__(f'llm_comparator.{module_name}', fromlist=[module_name])
            print(f"✅ {module_name}")
            available_modules.append(module_name)
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
    
    return available_modules

def test_error_taxonomy_if_available():
    """Test error taxonomy if it's available."""
    try:
        from llm_comparator import error_taxonomy
        
        print("\n🧪 Testing Error Taxonomy")
        print("=" * 40)
        
        taxonomy = error_taxonomy.ERROR_TAXONOMY
        print(f"Error types: {len(taxonomy.error_types)}")
        print(f"Categories: {len(taxonomy.get_all_categories())}")
        
        # Show first few error types
        print("\nSample error types:")
        for i, (key, error_type) in enumerate(list(taxonomy.error_types.items())[:3]):
            print(f"  {i+1}. {error_type.name}")
            print(f"     Category: {error_type.category.value}")
            print(f"     Severity: {error_type.severity_default.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error taxonomy test failed: {e}")
        return False

def main():
    print("🚀 LLM Comparator - Minimal Test")
    print("=" * 50)
    
    # Test what modules are available
    available = test_basic_imports()
    
    # Test error taxonomy if available
    if 'error_taxonomy' in available:
        success = test_error_taxonomy_if_available()
        if success:
            print("\n✅ Basic error taxonomy is working!")
            print("\n🎯 Next steps:")
            print("1. Apply the proposed changes for error_classifier.py")
            print("2. Apply the proposed changes for enhanced_comparison.py")
            print("3. Then run the full test suite")
        else:
            print("\n❌ Error taxonomy has issues")
    else:
        print("\n❌ Error taxonomy module not found")
        print("Check if the file was created properly")

if __name__ == "__main__":
    main()
