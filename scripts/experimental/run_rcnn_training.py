"""
Wrapper script to launch RCNN training with proper environment setup
"""
import os
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# Now run the training script
if __name__ == "__main__":
    # Import after path is set
    from scripts.train.rcnn_baseline import main
    
    # Set up command line arguments
    sys.argv = [
        'rcnn_baseline.py',
        '--epochs', '15',
        '--batch_size', '2', 
        '--lr', '0.0001',
        '--augment',
        '--optimizer', 'adamw',
        '--step_lr',
        '--device', 'cpu'
    ]
    
    main()
