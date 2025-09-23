import sys
import os
import torch
from torch.utils import checkpoint
from sklearn.model_selection import train_test_split

from srcv2full.engine import YAMNetEngine, trivial_collate_fn
from srcv2full.model import YAMNet
from srcv2full.logger import YamnetLogger
import srcv2full.params as params
from srcv2full.data import ESC50Data, ESC50ArtifactData

ep_epochs = 15  # Set the number of epochs for training

def validate_setup(dataset_path, log_path, checkpoint_path):
    """Validate that all paths and data are properly set up."""
    
    # Check dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Check required subdirectories
    audio_dir = os.path.join(dataset_path, "audio")
    meta_dir = os.path.join(dataset_path, "meta")
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    if not os.path.exists(meta_dir):
        raise FileNotFoundError(f"Meta directory not found: {meta_dir}")
    
    # Check metadata file
    meta_file = os.path.join(meta_dir, "esc50_artifact.csv")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created log directory: {log_dir}")
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    
    print("✓ All paths validated successfully")

def test_dataset_loading(dataset_path):
    """Test dataset loading and print basic info."""
    
    print("Testing dataset loading...")
    
    try:
        dataset = ESC50ArtifactData(dataset_path)
        print(f"✓ Dataset loaded successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of classes: {len(dataset.get_labels())}")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"  - Sample keys: {list(sample.keys())}")
        print(f"  - Audio shape: {sample['data'].shape}")
        print(f"  - Sample rate: {sample['sr']}")
        print(f"  - Artifact label: {sample['artifact_label']}")
        print(f"  - Base class: {sample['base_class']}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 example.py /path/to/dataset /path/to/log /path/to/checkpoint")
        print("Example: python3 example.py ESC50Artifact logs/yamnet.log checkpoints/yamnet.pth")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    log_path = sys.argv[2]
    checkpoint_path = sys.argv[3]
    
    print("YAMNet Training Setup")
    print("=" * 50)
    
    # Validate setup
    try:
        validate_setup(dataset_path, log_path, checkpoint_path)
        dataset = test_dataset_loading(dataset_path)
    except Exception as e:
        print(f"Setup validation failed: {e}")
        sys.exit(1)
    
    print("\nInitializing training components...")
    
    # Initialize logger
    log = YamnetLogger(log_path)
    log.info("Starting YAMNet training session")
    
    # Initialize model
    try:
        model = YAMNetEngine(
            model=YAMNet(),   
            # To use mbnv3 backbone: model=YAMNet(v3=True)
            tt_chunk_size=params.CHUNK_SIZE,
            logger=log,
        )
        log.info("✓ Model initialized successfully")
    except Exception as e:
        log.error(f"Model initialization failed: {e}")
        raise
    
    # Prepare data loaders
    try:
        train_idxs, val_idxs = train_test_split(
            range(len(dataset)), 
            test_size=0.2, 
            random_state=42,  # Fixed random state for reproducibility
            stratify=[dataset[i]["artifact_label"] for i in range(len(dataset))]  # Stratified split
        )
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idxs)
        
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            sampler=train_sampler, 
            num_workers=1, 
            collate_fn=trivial_collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            sampler=val_sampler, 
            num_workers=1, 
            collate_fn=trivial_collate_fn
        )
        
        log.info(f"✓ Data loaders created: {len(train_idxs)} train, {len(val_idxs)} val samples")
        
    except Exception as e:
        log.error(f"Data loader creation failed: {e}")
        raise
    
    # Start training
    try:
        log.info("Starting training process...")
        print(f"\nTraining Configuration:")
        print(f"  - Total epochs: {ep_epochs}")
        print(f"  - Learning rate: {params.LEARNING_RATE}")
        print(f"  - Chunk size: {params.CHUNK_SIZE}")
        print(f"  - Number of classes: {len(dataset.get_labels())}")
        print(f"  - Device: {model.device}")
        print(f"  - Checkpoint will be saved to: {checkpoint_path}")
        print("\nStarting training...\n")
        
        model.train_yamnet(
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_path=checkpoint_path,
            num_labels=len(dataset.get_labels()),
            num_epochs=ep_epochs
        )
        
        log.info("✓ Training completed successfully!")
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Model checkpoint saved to: {checkpoint_path}")
        print(f"Training logs saved to: {log_path}")
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")
        raise