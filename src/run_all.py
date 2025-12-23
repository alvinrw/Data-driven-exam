"""
Automated Runner - Greenhouse Control System

This script runs all files automatically in the correct order.
Suitable for regenerating all results from scratch.

WARNING: This script will run all time-consuming processes:
- Generate dataset: ~1 second
- Neural Network training: ~2-5 minutes
- MPC control: ~1-2 minutes
- Q-Learning training: ~5-10 minutes
- Comparison analysis: ~30 seconds

Total time: ~10-20 minutes
"""

import subprocess
import sys
import time
import os

print("=" * 70)
print("AUTOMATED RUNNER - GREENHOUSE CONTROL SYSTEM")
print("=" * 70)

# Daftar file yang akan dijalankan
scripts = [
    ("Generate Dataset", "genenratedata.py"),
    ("Train Neural Network", "NN_training.py"),
    ("Run MPC Control", "mpc_control.py"),
    ("Train Q-Learning", "qlearning_control.py"),
    ("Comparison Analysis", "comparison_analysis.py")
]

def run_script(name, filename):
    """Run a Python script and measure execution time"""
    print(f"\n{'=' * 70}")
    print(f"[{scripts.index((name, filename)) + 1}/{len(scripts)}] {name}")
    print(f"Running: {filename}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n[SUCCESS] {name} completed successfully!")
        print(f"  Execution time: {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n[FAILED] {name} failed!")
        print(f"  Execution time: {elapsed_time:.2f} seconds")
        print(f"  Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {filename}")
        return False

def main():
    """Main execution function"""
    
    # Change to the script's directory to ensure relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Confirmation
    print("\nThis script will run all greenhouse control algorithms.")
    print("Estimated total time: 10-20 minutes")
    print("\nFiles that will be executed:")
    for i, (name, filename) in enumerate(scripts, 1):
        print(f"  {i}. {filename} ({name})")
    
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nExecution cancelled.")
        return
    
    # Run all scripts
    total_start_time = time.time()
    results = []
    
    for name, filename in scripts:
        success = run_script(name, filename)
        results.append((name, success))
        
        if not success:
            print(f"\n[WARNING] {name} failed. Continuing with next script...")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\nExecution stopped by user.")
                break
    
    # Summary
    total_elapsed_time = time.time() - total_start_time
    
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    
    for name, success in results:
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"{status:12} - {name}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTotal: {successful}/{total} scripts completed successfully")
    print(f"Total execution time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    
    if successful == total:
        print("\n[COMPLETE] All scripts completed successfully!")
        print("\nGenerated files:")
        print("  - ../data/hfac_greenhouse_dataset.csv")
        print("  - ../models/hfac_model.h5, scaler_X.pkl, scaler_y.pkl")
        print("  - ../images/training_history.png, prediction_vs_actual.png")
        print("  - ../data/mpc_control_results.csv")
        print("  - ../images/mpc_control_results.png")
        print("  - ../models/qlearning_qtable.pkl")
        print("  - ../data/qlearning_control_results.csv, qlearning_training_history.csv")
        print("  - ../images/qlearning_control_results.png")
        print("  - ../data/algorithm_comparison.csv")
        print("  - ../images/algorithm_comparison.png")
    else:
        print("\n[WARNING] Some scripts failed. Please check the errors above.")

if __name__ == "__main__":
    main()
