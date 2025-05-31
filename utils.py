import numpy as np
import torch
from torch.utils.data import Dataset

def voxel_to_onehot(voxel, num_classes, block_id_to_class):
    """Convert a voxel grid from block IDs to one-hot encoding"""
    # voxel shape: (H, W, D)
    # output shape: (num_classes, H, W, D)
    H, W, D = voxel.shape
    onehot = np.zeros((num_classes, H, W, D), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            for k in range(D):
                block_id = voxel[i, j, k]
                # Ensure block_id is in the mapping to prevent KeyErrors
                if block_id in block_id_to_class:
                    class_idx = block_id_to_class[block_id]
                    onehot[class_idx, i, j, k] = 1.0
                # else: # Optional: handle unknown block_ids if necessary
                #     print(f"Warning: Block ID {block_id} not found in block_id_to_class mapping.")
    
    return onehot

class VoxelDatasetVAE(Dataset):
    def __init__(self, voxels_np, num_classes, block_id_to_class):
        self.voxels_np = voxels_np
        self.num_classes = num_classes
        self.block_id_to_class = block_id_to_class

    def __len__(self):
        return len(self.voxels_np)

    def __getitem__(self, idx):
        voxel_block_ids = self.voxels_np[idx]  # Shape: (H, W, D)
        
        # Convert to one-hot encoding
        voxel_onehot_np = voxel_to_onehot(
            voxel_block_ids, 
            self.num_classes, 
            self.block_id_to_class
        ) # Shape: (num_classes, H, W, D)
        
        voxel_onehot_torch = torch.tensor(voxel_onehot_np, dtype=torch.float32).contiguous()
        
        return voxel_onehot_torch
