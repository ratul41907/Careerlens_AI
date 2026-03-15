"""
Test Runner Script - Run all tests with coverage
"""
import subprocess
import sys


def run_tests():
    """Run all tests with pytest"""
    
    print("=" * 70)
    print("CAREERLENS AI - RUNNING ALL TESTS")
    print("=" * 70)
    
    # Run pytest with coverage
    cmd = [
        'pytest',
        'tests/',
        '-v',  # Verbose
        '--cov=src',  # Coverage for src directory
        '--cov-report=html',  # HTML report
        '--cov-report=term',  # Terminal report
        '-m', 'not slow',  # Skip slow tests
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n📊 Coverage report generated in: htmlcov/index.html")
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    run_tests()