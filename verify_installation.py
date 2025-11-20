#!/usr/bin/env python3
"""
Verification script to test ConsistencyAI installation.

Run this after installation to ensure everything is working correctly.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import duplicity
        print("   Main package imported")
    except ImportError as e:
        print(f"   Failed to import duplicity: {e}")
        return False
    
    # Test core modules
    modules = [
        "llm_tool",
        "personas",
        "queries",
        "fast_robust_queries",
        "similarity",
        "visualization",
        "embedding_analysis",
        "config"
    ]
    
    for module in modules:
        try:
            exec(f"from duplicity import {module}")
            print(f"   {module} imported")
        except ImportError as e:
            print(f"   Failed to import {module}: {e}")
            return False
    
    return True


def test_core_functions():
    """Test that core functions are accessible."""
    print("\nTesting core functions...")
    
    functions = [
        "get_and_clean_personas",
        "generate_queries_for_personas",
        "query_llm_fast",
        "supercompute_similarities",
        "plot_overall_leaderboard",
        "analyze_and_cluster_embeddings",
    ]
    
    try:
        from duplicity import (
            get_and_clean_personas,
            generate_queries_for_personas,
            query_llm_fast,
            supercompute_similarities,
            plot_overall_leaderboard,
            analyze_and_cluster_embeddings,
        )
        
        for func in functions:
            print(f"   {func} accessible")
        
        return True
    except ImportError as e:
        print(f"   Failed to import functions: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nTesting dependencies...")
    
    dependencies = [
        "aiohttp",
        "requests",
        "numpy",
        "pandas",
        "sklearn",
        "sentence_transformers",
        "matplotlib",
        "seaborn",
        "torch",
        "transformers",
        "plotly",
        "tqdm",
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   {dep} installed")
        except ImportError:
            print(f"   {dep} NOT installed")
            all_ok = False
    
    return all_ok


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    try:
        from duplicity import config
        
        # Check if API key is set (don't print it)
        if config.OPENROUTER_API_KEY:
            print("   OPENROUTER_API_KEY is set")
        else:
            print("   OPENROUTER_API_KEY is not set")
            print("    Set it with: export OPENROUTER_API_KEY='your-key'")
            print("    Or in Python: config.set_openrouter_key('your-key')")
        
        # Test config functions
        print("   Configuration module accessible")
        return True
    except Exception as e:
        print(f"   Configuration error: {e}")
        return False


def test_version():
    """Test version information."""
    print("\nVersion information...")
    
    try:
        from duplicity import __version__, __author__
        print(f"  ConsistencyAI version: {__version__}")
        print(f"  Authors: {__author__}")
        return True
    except Exception as e:
        print(f"   Failed to get version info: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ConsistencyAI Installation Verification")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_core_functions,
        test_dependencies,
        test_config,
        test_version,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n   Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("SUCCESS: All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Set your API key if not already done:")
        print("   export OPENROUTER_API_KEY='your-key'")
        print("2. Read the documentation:")
        print("   - README.md for overview")
        print("   - docs/QUICKSTART.md for quick start")
        print("   - docs/API.md for complete API reference")
        print("3. Try the main.ipynb notebook or build your own experiment")
        return 0
    else:
        print("FAILED: Some tests did not pass")
        print("=" * 60)
        print("\nPlease check the errors above and:")
        print("1. Ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("2. Install the package:")
        print("   pip install -e .")
        print("3. Run this script again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
