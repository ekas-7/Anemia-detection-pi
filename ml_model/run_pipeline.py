"""
Complete ML Model Pipeline
Runs the entire pipeline from data loading to model training
"""

import subprocess
import sys
from pathlib import Path
import time


def run_command(command, description, cwd=None):
    """
    Run a command and handle errors
    
    Args:
        command: Command to run
        description: Description of the step
        cwd: Working directory
    """
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Command: {command}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error in {description}: {e}")
        return False


def main():
    """
    Run the complete pipeline
    """
    print("="*60)
    print("ML MODEL - COMPLETE PIPELINE")
    print("="*60)
    print("\nThis script will:")
    print("1. Download the Eyes Defy Anemia dataset")
    print("2. Preprocess and augment the data")
    print("3. Extract color and morphological features")
    print("4. Train multiple ML models")
    print("5. Generate comprehensive evaluation metrics")
    print("\nEstimated time: 30-60 minutes (depending on dataset size)")
    print("="*60)
    
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled.")
        return
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Pipeline steps
    steps = [
        {
            'command': f'{sys.executable} data/load_dataset.py',
            'description': 'Download and load dataset',
            'cwd': project_root
        },
        {
            'command': f'{sys.executable} data/preprocess.py',
            'description': 'Preprocess and augment images',
            'cwd': project_root
        },
        {
            'command': f'{sys.executable} features/extract_features.py',
            'description': 'Extract features',
            'cwd': project_root
        },
        {
            'command': f'{sys.executable} training/train_ml_model.py',
            'description': 'Train and evaluate models',
            'cwd': project_root
        }
    ]
    
    # Run pipeline
    start_time = time.time()
    failed_steps = []
    
    for i, step in enumerate(steps, 1):
        print(f"\n\n{'#'*60}")
        print(f"PIPELINE STEP {i}/{len(steps)}")
        print(f"{'#'*60}")
        
        success = run_command(
            step['command'],
            step['description'],
            step['cwd']
        )
        
        if not success:
            failed_steps.append(step['description'])
            print(f"\n✗ Pipeline failed at step {i}: {step['description']}")
            print("Please check the errors above and try again.")
            return
        
        # Small delay between steps
        time.sleep(2)
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nTotal time: {total_time/60:.2f} minutes")
    print("\nResults:")
    print(f"  ✓ Trained models saved in: {project_root / 'models'}")
    print(f"  ✓ Training results: {project_root / 'models' / 'training_results.json'}")
    print(f"  ✓ Visualizations: {project_root / 'models'}/*.png")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Review training results in models/training_results.json")
    print("2. Check visualizations: confusion matrices and metrics comparison")
    print("3. Test inference:")
    print(f"   python inference/predict.py --image <path_to_image> --compare")
    print("4. Deploy the best model to your mobile/embedded device")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
