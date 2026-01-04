#!/usr/bin/env python3
"""
Verify Import Chain - Dataset Creator Workflow
Tests that all imports in the workflow chain are valid
"""

import sys
from pathlib import Path

def test_import(module_path, description):
    """Test if a module can be imported"""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Module: {module_path}")
    print(f"{'='*80}")
    
    try:
        # Try to import the module
        if '.' in module_path:
            parts = module_path.split('.')
            module = __import__(module_path)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(module_path)
        
        print(f"✅ SUCCESS: {description} imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: {description}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  WARNING: {description}")
        print(f"   Error: {e}")
        return False


def main():
    """Test the complete import chain"""
    print("\n" + "="*80)
    print("DATASET CREATOR IMPORT CHAIN VERIFICATION")
    print("="*80)
    
    results = []
    
    # Step 1: Project Dashboard (Entry Point)
    results.append(test_import(
        "src.core.dataset_creator.project_management.project_dashboard",
        "Project Dashboard (Entry Point)"
    ))
    
    # Step 2: Project Manager
    results.append(test_import(
        "src.core.dataset_creator.project_management.project_manager",
        "Project Manager"
    ))
    
    # Step 3: Enterprise Dataset Studio
    results.append(test_import(
        "src.core.dataset_creator.enterprise_dataset_studio",
        "Enterprise Dataset Studio"
    ))
    
    # Step 4: Components
    results.append(test_import(
        "src.core.dataset_creator.components.professional_dataset_selector",
        "Professional Dataset Selector"
    ))
    
    results.append(test_import(
        "src.core.dataset_creator.components.pipeline_compatibility_engine",
        "Pipeline Compatibility Engine"
    ))
    
    # Step 5: Dataset Frame (Core Studio)
    results.append(test_import(
        "src.core.dataset_creator.truscore_dataset_frame_flowlayout",
        "TruScore Dataset Frame (FlowLayout)"
    ))
    
    # Step 6: FlowLayout
    results.append(test_import(
        "src.core.dataset_creator.flowlayout",
        "FlowLayout (Grid System)"
    ))
    
    # Step 7: Utilities
    results.append(test_import(
        "src.core.dataset_creator.yolo_to_maskrcnn_converter",
        "YOLO to Mask R-CNN Converter"
    ))
    
    results.append(test_import(
        "src.core.dataset_creator.project_management.label_pipeline_compatibility",
        "Label Pipeline Compatibility"
    ))
    
    results.append(test_import(
        "src.core.dataset_creator.enterprise_glassmorphism",
        "Enterprise Glassmorphism (UI Styling)"
    ))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {total - passed} ❌")
    
    if passed == total:
        print("\n🎉 ALL IMPORTS VALID! The workflow should work correctly!")
    else:
        print(f"\n⚠️  {total - passed} import(s) failed. Fix these before running the workflow.")
    
    print("="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
