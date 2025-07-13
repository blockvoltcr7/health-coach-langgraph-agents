#!/usr/bin/env python3
"""Run sales process tracking tests."""

import subprocess
import sys

def run_tests():
    """Run the sales process tracking tests."""
    print("Running sales process tracking tests...")
    print("-" * 60)
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, 
        "-m", 
        "pytest", 
        "tests/test_sales_process_tracking.py",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())