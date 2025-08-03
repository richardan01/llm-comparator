#!/usr/bin/env python3
"""Simple test script that works with existing files only."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from llm_comparator import error_taxonomy
    print("✅ Successfully imported error_taxonomy")
    
    # Test the error taxonomy
    print("\n🧪 Testing Error Taxonomy System")
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
    
    print("\n✅ Error taxonomy test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Let's check what files exist...")
    
    src_dir = os.path.join(os.path.dirname(__file__), 'src', 'llm_comparator')
    if os.path.exists(src_dir):
        files = os.listdir(src_dir)
        print(f"Files in llm_comparator directory: {files}")
    else:
        print(f"Directory not found: {src_dir}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()