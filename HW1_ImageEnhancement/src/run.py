import os
import subprocess
import sys

def run(script_path):
    # Run a Python script as a subprocess and checks for errors
    print(f"\n{'='*50}")
    print(f"Running: {script_path}")
    print(f"{'='*50}")

    # Ensure that we use the same Python interpreter
    result = subprocess.run([sys.executable, script_path], capture_output=False)

    if result.returncode != 0:
        print(f"[ERROR] Script {script_path} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[SUCCESS] {script_path} completed")

def main():
    scripts = ["src/degrade.py", "src/visualize.py", "src/evaluate.py"]

    for script in scripts:
        if not os.path.exists(script):
            print(f"[ERROR] Script not found: {script}")
            sys.exit(1)
        run(script)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check 'results/images/' for visual comparisons.")
    print("Check 'results/metrics/metrics.csv' for the evaluation table.")
    print("="*50)

if __name__ == "__main__":
    main()