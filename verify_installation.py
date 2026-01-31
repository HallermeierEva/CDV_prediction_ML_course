"""
Installation Verification Script
Author: Eva Hallermeier

This script verifies that all dependencies are installed correctly
and the project structure is set up properly.

Usage:
    python verify_installation.py
"""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (incompatible)")
        print("  Please use Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy'
    }
    
    all_installed = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (not installed)")
            all_installed = False
    
    if not all_installed:
        print("\n  Install missing packages with:")
        print("  pip install -r requirements.txt")
    
    return all_installed

def check_project_structure():
    """Check if project directories exist."""
    print("\nChecking project structure...")
    
    required_dirs = [
        'data',
        'src',
        'results',
        'results/figures',
        'results/models',
        'results/metrics'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False
            # Try to create it
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  → Created {dir_path}/")
            except Exception as e:
                print(f"  → Failed to create: {e}")
    
    return all_exist

def check_data_file():
    """Check if the dataset exists."""
    print("\nChecking data file...")
    data_file = Path('data/cardio_dataset.csv')
    
    if data_file.exists():
        file_size = data_file.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"✓ cardio_dataset.csv ({file_size:.2f} MB)")
        return True
    else:
        print(f"✗ cardio_dataset.csv (not found)")
        print("  Expected location: data/cardio_dataset.csv")
        print("  Please ensure the dataset is in the data directory")
        return False

def check_source_files():
    """Check if all source files exist."""
    print("\nChecking source files...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/config.py',
        'src/preprocessing.py',
        'src/baseline_model.py',
        'src/xgboost_model.py',
        'src/visualization.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def test_import():
    """Test importing the project modules."""
    print("\nTesting module imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src import config
        print("✓ src.config")
        
        from src import preprocessing
        print("✓ src.preprocessing")
        
        from src import baseline_model
        print("✓ src.baseline_model")
        
        from src import xgboost_model
        print("✓ src.xgboost_model")
        
        from src import visualization
        print("✓ src.visualization")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("="*80)
    print("CVD PREDICTION PROJECT - INSTALLATION VERIFICATION")
    print("="*80)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Project Structure': check_project_structure(),
        'Data File': check_data_file(),
        'Source Files': check_source_files(),
        'Module Imports': test_import()
    }
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:<20} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to run the project.")
        print("\nNext step:")
        print("  python main.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Ensure cardio_dataset.csv is in the data/ directory")
        print("  3. Verify you're in the project root directory")
        return 1

if __name__ == '__main__':
    sys.exit(main())
