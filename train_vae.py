import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import os
import gc
from vae_model import VAE  # Our VAE model
from utils import VoxelDatasetVAE # Our new dataset and utility

# ========== Configuration ==========
# Explicitly select GPU 1 which is free
torch.cuda.set_device(1)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATA_PATH = 'schematics_with_labels.npz' # Smaller dataset with 6K samples
NUM_EPOCHS = 50 # Increased since we have fewer samples
BATCH_SIZE = 8 # Can use larger batch with smaller dataset
LEARNING_RATE = 1e-4
LATENT_CHANNELS = 32 # Must match the VAE definition
KLD_BETA = 0.00025 # Weight for KL divergence
CHECKPOINT_DIR = "vae_checkpoints"
SAVE_EVERY_N_EPOCHS = 1
NUM_WORKERS = 2 # Use workers for faster data loading
PREFETCH_FACTOR = 1 # Increased for better throughput

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========== Loss Function ==========
def vae_loss_function(recon_x_logits, x_onehot, mu, log_var, beta):
    # recon_x_logits shape: (B, C, D, H, W) where C is num_classes
    # x_onehot shape: (B, C, D, H, W)
    
    # Reconstruction Loss (Cross-Entropy)
    # Target for cross_entropy should be class indices: (B, D, H, W)
    target_indices = x_onehot.argmax(dim=1) 
    
    recon_loss = F.cross_entropy(
        recon_x_logits, # Expected (N, C, d1, d2, ...)
        target_indices, # Expected (N, d1, d2, ...)
        reduction='sum' 
    ) / x_onehot.shape[0] # Average over batch size

    # KL Divergence
    # mu, log_var shape: (B, latent_channels, LD, LH, LW)
    # We want to sum over all dimensions except batch, then average over batch.
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=list(range(1, mu.dim())))
    kld_loss = torch.mean(kld_loss) # Average over batch

    return recon_loss + beta * kld_loss, recon_loss, kld_loss

# ========== Main Training Script ==========
def main():
    print(f"Using device: {DEVICE}")

    # --- 1. Load Dataset ---
    print("Loading dataset...")
    # Use memory mapping for initial load to reduce memory usage
    data_content = np.load(DATA_PATH, mmap_mode='r')
    # Create a copy of just the voxels array we need
    voxels_np = data_content['voxels'].copy()  # Shape (N, H, W, D) - block IDs
    print(f"voxels_np shape: {voxels_np.shape}")
    print(f"voxels_np dtype: {voxels_np.dtype}")
    print(f"voxels_np nbytes: {voxels_np.nbytes / (1024**3):.2f} GB")

    # Explicitly delete the loaded .npz data object and collect garbage
    del data_content

    gc.collect()
    print("data_content deleted and memory cleaned up.")

    # Load pre-calculated block info
    print("Loading pre-calculated block info from block_info.npz...")
    try:
        block_info = np.load('block_info.npz', allow_pickle=True)
        num_block_types = int(block_info['num_block_types'])
        block_id_to_class = block_info['block_id_to_class'].item() # .item() to get dict from 0-d array
        print(f"Loaded {num_block_types} unique block types and block_id_to_class mapping.")
    except FileNotFoundError:
        print("ERROR: block_info.npz not found. Please run preprocess_block_info.py first.")
        return
    except Exception as e:
        print(f"Error loading block_info.npz: {e}")
        return

    vae_dataset = VoxelDatasetVAE(voxels_np, num_block_types, block_id_to_class)
    # Memory-optimized DataLoader configuration based on previous successful runs
    train_loader = DataLoader(
        vae_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=False,  # Must be False to avoid CUDA errors with this dataset
        prefetch_factor=PREFETCH_FACTOR
    )
    print(f"Dataset loaded. Number of samples: {len(vae_dataset)}")
    
    # Free memory after dataset creation
    del voxels_np
    gc.collect()

    # --- 2. Initialize Model ---
    print("Initializing VAE model...")
    model = VAE(input_channels=num_block_types, latent_channels=LATENT_CHANNELS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model and optimizer initialized.")

    # --- 3. Training Loop ---
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss_epoch = 0
        total_recon_loss_epoch = 0
        total_kld_loss_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for batch_idx, voxel_onehot in enumerate(progress_bar):
            voxel_onehot = voxel_onehot.to(DEVICE) # Shape (B, num_classes, D, H, W)
            
            optimizer.zero_grad()
            
            recon_voxels_logits, mu, log_var = model(voxel_onehot)
            
            loss, recon_loss, kld_loss = vae_loss_function(
                recon_voxels_logits, voxel_onehot, mu, log_var, KLD_BETA
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            total_recon_loss_epoch += recon_loss.item()
            total_kld_loss_epoch += kld_loss.item()
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4e}", 
                "ReconL": f"{recon_loss.item():.4e}", 
                "KLDL": f"{kld_loss.item():.4e}"
            })
            
            # Free up memory after each batch
            del voxel_onehot, recon_voxels_logits, mu, log_var, loss, recon_loss, kld_loss

        avg_loss = total_loss_epoch / len(train_loader)
        avg_recon_loss = total_recon_loss_epoch / len(train_loader)
        avg_kld_loss = total_kld_loss_epoch / len(train_loader)
        print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg KLD: {avg_kld_loss:.4f}")

        # Save checkpoint
        # Clean up memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()
        
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"vae_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'num_block_types': num_block_types, 
                'latent_channels': LATENT_CHANNELS,
                'block_id_to_class': block_id_to_class, # Save for inference
                'KLD_BETA': KLD_BETA
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training finished.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()
        print("Training completed or interrupted. Memory cleaned up.")
