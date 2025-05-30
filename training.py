import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import CLIPTokenizer, CLIPTextModel

from tqdm import tqdm

import math

from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

# import argparse # Reverted: Removed argparse import



# ========== Sinusoidal Positional Embedding for Timesteps ==========
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: 1D tensor of timesteps
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:  # Zero pad if dim is odd
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings



# ========== Self-Attention Module ==========
class SelfAttention3D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.query = nn.Conv3d(channels, channels, 1)
        self.key = nn.Conv3d(channels, channels, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.out_conv = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling factor

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        q = self.query(x).view(B, self.num_heads, C // self.num_heads, D * H * W)
        k = self.key(x).view(B, self.num_heads, C // self.num_heads, D * H * W)
        v = self.value(x).view(B, self.num_heads, C // self.num_heads, D * H * W)

        q = q.permute(0, 1, 3, 2)  # (B, num_heads, D*H*W, C//num_heads)
        # k is (B, num_heads, C//num_heads, D*H*W)
        # v is (B, num_heads, C//num_heads, D*H*W)

        attention_scores = torch.matmul(q, k) * ((C // self.num_heads) ** -0.5) # Scale dot product
        attention_probs = F.softmax(attention_scores, dim=-1) # (B, num_heads, D*H*W, D*H*W)

        attention_output = torch.matmul(attention_probs, v.permute(0,1,3,2)) # (B, num_heads, D*H*W, C//num_heads)
        attention_output = attention_output.permute(0,1,3,2).contiguous().view(B, C, D, H, W)
        
        out = self.out_conv(attention_output)
        return x + self.gamma * out # Residual connection with learnable scale



# ========== GPU Setup ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():

    # Set GPU memory growth

    torch.backends.cudnn.benchmark = True

    # Optional: Set specific GPU if you have multiple

    # torch.cuda.set_device(0)

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

else:

    print("Using CPU")



# ========== Load Dataset ==========

data = np.load('schematics_with_labels.npz', allow_pickle=True)

voxels_np = data['voxels']

labels_np = data['labels']



# ========== Analyze Dataset for One-Hot Encoding ==========

print("Analyzing dataset for unique block types...")

unique_blocks = np.unique(voxels_np)

num_block_types = len(unique_blocks)

print(f"Found {num_block_types} unique block types: {unique_blocks}")



# Create mapping from block ID to class index

block_id_to_class = {block_id: idx for idx, block_id in enumerate(unique_blocks)}

class_to_block_id = {idx: block_id for idx, block_id in enumerate(unique_blocks)}



print("Block ID to Class mapping:")

for block_id, class_idx in list(block_id_to_class.items())[:10]:  # Show first 10

    print(f"  Block ID {block_id} -> Class {class_idx}")

if len(block_id_to_class) > 10:

    print(f"  ... and {len(block_id_to_class) - 10} more mappings")



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

                class_idx = block_id_to_class[block_id]

                onehot[class_idx, i, j, k] = 1.0

    

    return onehot



def onehot_to_voxel(onehot, class_to_block_id):

    """Convert one-hot encoding back to block IDs"""

    # onehot shape: (num_classes, H, W, D)

    # output shape: (H, W, D)

    class_indices = np.argmax(onehot, axis=0)  # Get most likely class for each voxel

    

    H, W, D = class_indices.shape

    voxel = np.zeros((H, W, D), dtype=np.uint8)

    

    for i in range(H):

        for j in range(W):

            for k in range(D):

                class_idx = class_indices[i, j, k]

                block_id = class_to_block_id[class_idx]

                voxel[i, j, k] = block_id

    

    return voxel



# ========== Text Embedding ==========

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

text_encoder.eval()



# Move to GPU if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder = text_encoder.to(device)



# ========== Diffusion Scheduler ==========

def get_beta_schedule(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):

    """Linear beta schedule for diffusion."""

    return torch.linspace(beta_start, beta_end, num_timesteps)



def get_alpha_schedule(betas):

    """Compute alpha values from beta schedule."""

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return alphas, alphas_cumprod



# Initialize diffusion parameters

betas = get_beta_schedule()

alphas, alphas_cumprod = get_alpha_schedule(betas)



# ========== Diffusion Parameters ==========
num_timesteps = 1000  # Should match betas length
# Ensure betas, alphas, alphas_cumprod are on CPU for broader compatibility initially
# They will be moved to device in 'extract' or when used.
betas = betas.cpu()
alphas = alphas.cpu()
alphas_cumprod = alphas_cumprod.cpu()

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# For q(x_{t-1} | x_t, x_0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
# Clamp variance to prevent log(0) if posterior_variance is zero
posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

# Coefficients for posterior mean q(x_{t-1} | x_t, x_0)
# mu_tilde_t(x_t, x_0) = coef1 * x_0 + coef2 * x_t
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

def extract(a, t, x_shape):
    """Extracts values from a tensor 'a' at given timesteps 't' and reshapes them for broadcasting."""
    batch_size = t.shape[0]
    # Ensure 'a' is on the same device as 't' before gather
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# ========== Dataset ==========

class VoxelDataset(Dataset):

    def __init__(self, voxels, labels, tokenizer, num_classes, block_id_to_class):

        self.voxels = voxels

        self.labels = labels

        self.tokenizer = tokenizer

        self.num_classes = num_classes

        self.block_id_to_class = block_id_to_class



    def __len__(self):

        return len(self.voxels)



    def __getitem__(self, idx):

        voxel_block_ids = self.voxels[idx]  # These are block IDs (H, W, D)
        
        # Convert to one-hot encoding
        voxel_onehot_np = voxel_to_onehot(
            voxel_block_ids, 
            self.num_classes, 
            self.block_id_to_class
        ) # Shape: (num_classes, H, W, D)
        
        voxel_onehot_torch = torch.tensor(voxel_onehot_np, dtype=torch.float32)
        
        prompt = self.labels[idx]
        
        # Proper tokenization with max_length to ensure consistent dimensions
        tokenized = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=77  # CLIP's standard max length
        )
        return voxel_onehot_torch, tokenized



# ========== Improved 3D U-Net for One-Hot Encoding ==========

class Simple3DUNet(nn.Module):

    def __init__(self, num_classes, text_embed_dim=512, time_embed_dim=128):

        super().__init__()

        self.num_classes = num_classes

        num_groups = 8 # For GroupNorm
        

        # Time embedding

        self.time_mlp = SinusoidalPositionalEmbedding(time_embed_dim)

        

        # Encoder

        self.down1_conv = nn.Conv3d(num_classes, 32, 3, padding=1)

        self.down1_gn = nn.GroupNorm(num_groups, 32)

        

        self.down2_conv = nn.Conv3d(32, 64, 3, stride=2, padding=1)  # 32x32x32

        self.down2_gn = nn.GroupNorm(num_groups, 64)

        

        self.down3_conv = nn.Conv3d(64, 128, 3, stride=2, padding=1)  # 16x16x16

        self.down3_gn = nn.GroupNorm(num_groups, 128)

        

        # Bottleneck

        self.mid_conv = nn.Conv3d(128, 128, 3, padding=1)

        self.mid_gn = nn.GroupNorm(num_groups, 128)

        self.mid_attn = SelfAttention3D(128, num_heads=4) # Add attention layer

        

        # Decoder

        self.up1_convt = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)  # 32x32x32

        self.up1_gn = nn.GroupNorm(num_groups, 64)

        

        self.up2_convt = nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1)   # 64x64x64

        self.up2_gn = nn.GroupNorm(num_groups, 32)

        

        self.out_conv = nn.Conv3d(32, num_classes, 3, padding=1)  # Output logits for each class

        

        # Conditioning

        self.text_embed = nn.Sequential(

            nn.Linear(text_embed_dim, time_embed_dim),

            nn.ReLU(),

            nn.Linear(time_embed_dim, time_embed_dim)

        )

        

        # Conditioning injection layers
        # Ensure output dimensions match the number of channels in the U-Net layers
        self.cond_proj_down1 = nn.Linear(time_embed_dim, 32)
        self.cond_proj_down2 = nn.Linear(time_embed_dim, 64)
        self.cond_proj_down3_mid = nn.Linear(time_embed_dim, 128) # For down3 and mid
        self.cond_proj_up1 = nn.Linear(time_embed_dim, 64) # For up1 output (matches h2 channels)
        self.cond_proj_up2 = nn.Linear(time_embed_dim, 32) # For up2 output (matches h1 channels)



    def forward(self, x, t, text_embed):
        # Process conditioning
        # t is (B,) tensor of timesteps
        t_embed = self.time_mlp(t)      # (B, time_embed_dim)
        txt_embed = self.text_embed(text_embed)    # (B, time_embed_dim)
        cond = t_embed + txt_embed                 # (B, time_embed_dim)
        
        # Get conditioning projections, unsqueezed for broadcasting
        cond_d1 = self.cond_proj_down1(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_d2 = self.cond_proj_down2(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_d3_mid = self.cond_proj_down3_mid(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_u1 = self.cond_proj_up1(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_u2 = self.cond_proj_up2(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Encoder
        x1 = F.relu(self.down1_gn(self.down1_conv(x)))
        x1_c = x1 + cond_d1 # Apply conditioning after activation of the block
        
        x2 = F.relu(self.down2_gn(self.down2_conv(x1_c)))
        x2_c = x2 + cond_d2
        
        x3 = F.relu(self.down3_gn(self.down3_conv(x2_c)))
        x3_c = x3 + cond_d3_mid
        
        # Bottleneck
        h_mid_features = self.mid_gn(self.mid_conv(x3_c))
        h_mid_activated = F.relu(h_mid_features)
        h_mid_attended = self.mid_attn(h_mid_activated) # Apply attention
        h_mid_c = h_mid_attended + cond_d3_mid # Re-apply/add conditioning to bottleneck output
        
        # Decoder with skip connections
        h_up1 = F.relu(self.up1_gn(self.up1_convt(h_mid_c)))
        h_up1_cat = h_up1 + x2_c # Skip connection from corresponding encoder stage (after its conditioning)
        h_up1_cond = h_up1_cat + cond_u1 # Apply decoder block's conditioning
        
        h_up2 = F.relu(self.up2_gn(self.up2_convt(h_up1_cond)))
        h_up2_cat = h_up2 + x1_c # Skip connection
        h_up2_cond = h_up2_cat + cond_u2 # Apply decoder block's conditioning
        
        # Output logits (no activation - will use with cross-entropy loss)
        return self.out_conv(h_up2_cond)



# ========== Categorical Diffusion Training ==========
def q_sample(x_start_onehot, t):
    """
    Noises x_start (one-hot) to x_t.
    For categorical diffusion, we typically noise the *probabilities* or *logits*.
    Here, we'll noise the one-hot encoding, which can be seen as probabilities
    where one class is 1 and others are 0. The model will then predict the
    original class logits.
    """
    noise = torch.randn_like(x_start_onehot) # Noise has same shape as one-hot input
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start_onehot.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start_onehot.shape)
    
    # Noisy version of x_start_onehot
    x_t = sqrt_alphas_cumprod_t * x_start_onehot + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise # Return noise for potential direct prediction, though we predict x0

def train_categorical(model, loader, optimizer, text_encoder, num_classes, epochs=5, class_weights=None, scheduler=None):
    model.train()
    global_step = 0
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = len(loader)
        
        # Progress bar for batches
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", 
                         unit="batch", leave=False)
        
        for batch_idx, (voxels_onehot, tokenized) in enumerate(batch_pbar):
            # Clear GPU cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            x_start_onehot = voxels_onehot.to(device)  # (B, num_classes, H, W, D)
            batch_size = x_start_onehot.size(0)
            
            # Sample timesteps t for diffusion
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            
            # Create x_t by noising x_0 (x_start_onehot)
            # The q_sample function handles the noising based on DDPM schedule
            x_t, _ = q_sample(x_start_onehot, t) # Removed num_classes argument
            
            # Get text embeddings
            with torch.no_grad():
                text_inputs = {k: v.squeeze(1).to(device) for k, v in tokenized.items()}
                text_embed = text_encoder(**text_inputs).pooler_output
            
            # Mixed precision training
            with autocast():
                # Model predicts logits for x_0 given x_t and t
                pred_x0_logits = model(x_t, t, text_embed) 
                
                # Target for cross-entropy is the original class indices
                target_classes = torch.argmax(x_start_onehot, dim=1)  # (B, H, W, D)
                
                # Compute cross-entropy loss between predicted logits and true classes
                # pred_x0_logits shape: (B, num_classes, H, W, D)
                # target_classes shape: (B, H, W, D)
                loss = F.cross_entropy(pred_x0_logits, target_classes)
            
            # Backpropagation with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar with GPU memory info
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}',
                    'GPU Memory': f'{gpu_memory:.2f}GB'
                })
            else:
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()

    # Save the final trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'block_id_to_class': block_id_to_class,
        'class_to_block_id': class_to_block_id,
    }, 'voxel_categorical.pth')
    print("Final model saved to voxel_categorical.pth")



# ========== Categorical Sampling ==========
def p_sample_loop(model, text_embed, shape, num_classes, class_to_block_id, steps=None):
    """Ancestral sampling loop for DDPM."""
    model.eval()
    
    if steps is None:
        steps = num_timesteps # Use all diffusion timesteps for sampling

    # Start with random Gaussian noise (x_T)
    # Shape: (1, num_classes, H, W, D) - matching model input
    x_t = torch.randn(shape, device=device) 
    
    sampling_pbar = tqdm(reversed(range(steps)), desc="DDPM Sampling", total=steps, unit="step")

    for i in sampling_pbar:
        t_batch = torch.full((shape[0],), i, device=device, dtype=torch.long) # Batch of timesteps

        with torch.no_grad():
            # Model predicts logits for x_0
            pred_x0_logits = model(x_t, t_batch, text_embed)
            # Convert predicted logits to probabilities for x_0
            pred_x0_probs = F.softmax(pred_x0_logits, dim=1) # (B, num_classes, H, W, D)

            if i == 0: # If t is 0, this is the final denoised sample
                x_t = pred_x0_probs # The model's best guess for x_0 (as probabilities)
                break

            # Calculate posterior mean and variance for q(x_{t-1} | x_t, x_0_hat=pred_x0_probs)
            posterior_mean_coef1_t = extract(posterior_mean_coef1, t_batch, x_t.shape)
            posterior_mean_coef2_t = extract(posterior_mean_coef2, t_batch, x_t.shape)
            
            posterior_mean = posterior_mean_coef1_t * pred_x0_probs + posterior_mean_coef2_t * x_t
            
            posterior_log_variance_t = extract(posterior_log_variance_clipped, t_batch, x_t.shape)
            
            noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t) # No noise at the last step
            x_t = posterior_mean + (0.5 * posterior_log_variance_t).exp() * noise
            
    # x_t now holds the probabilities of x_0
    # Final step: convert probabilities to discrete classes by taking argmax
    # x_t shape: (1, num_classes, H, W, D)
    sampled_classes = torch.argmax(x_t, dim=1).squeeze(0) # Squeeze batch dim -> (H, W, D)
    
    # Convert back to numpy
    sampled_classes_np = sampled_classes.cpu().numpy().astype(np.uint8)
    
    # Convert class indices back to block IDs
    H_dim, W_dim, D_dim = sampled_classes_np.shape
    voxel_result = np.zeros((H_dim, W_dim, D_dim), dtype=np.uint8)
    for r_idx in range(H_dim):
        for c_idx in range(W_dim):
            for d_idx in range(D_dim):
                class_idx = sampled_classes_np[r_idx, c_idx, d_idx]
                voxel_result[r_idx, c_idx, d_idx] = class_to_block_id[int(class_idx)]
    
    return voxel_result


def sample_categorical(text_prompt, model, text_encoder, tokenizer, num_classes, class_to_block_id, steps=None):
    model.eval()
    
    # Tokenize text prompt
    text_tokens = tokenizer(
        text_prompt, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True, 
        max_length=77
    )
    
    # Get text embedding
    with torch.no_grad():
        text_inputs = {k: v.to(device) for k, v in text_tokens.items()}
        text_embed = text_encoder(**text_inputs).pooler_output # Shape (1, embed_dim)
    
    # Define the shape of the voxel grid to generate
    # Assuming H, W, D are 64 based on previous context of x in sample_categorical
    generated_shape = (1, num_classes, 64, 64, 64) 
    
    return p_sample_loop(model, text_embed, generated_shape, num_classes, class_to_block_id, steps)



# ========== Main ==========

if __name__ == "__main__":
    # Reverted: Removed argparse setup

    # Defaults (restored)
    epochs = 50
    prompt = "House"
    checkpoint_path = "voxel_categorical.pth"
    output_path = "generated_voxel.npy"

    print(f"Using device: {device}")
    print(f"Dataset size: {len(voxels_np)} voxels")

    # Initialize model
    model = Simple3DUNet(num_classes=num_block_types).to(device)

    # ==== Training Phase ====
    print(f"Starting training for {epochs} epochs...")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # T_max for scheduler should match the number of epochs used for training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) 
    
    dataset = VoxelDataset(voxels_np, labels_np, tokenizer, num_classes=num_block_types, block_id_to_class=block_id_to_class)
    loader = DataLoader(
        dataset, 
        batch_size=4,  
        shuffle=True, 
        num_workers=2,  
        pin_memory=True  
    )
    
    print(f"Training with batch size: {loader.batch_size}")
    print(f"Number of batches per epoch: {len(loader)}")
    
    # To run inference only, comment out the following line:
    train_categorical(model, loader, optimizer, text_encoder, num_classes=num_block_types, epochs=epochs, scheduler=scheduler)
    
    print(f"Training finished. Model saved to {checkpoint_path}")
    
    # Ensure model is the base model if DataParallel was used, for consistent inference loading
    if isinstance(model, nn.DataParallel):
        model = model.module

    # ==== Inference Phase ====
    print(f"Loading model from {checkpoint_path} for inference...")
    
    # It's good practice to initialize a new model instance for inference, 
    # especially if you might comment out the training section entirely.
    try:
        # Load checkpoint first to get num_classes if it was saved
        checkpoint = torch.load(checkpoint_path, map_location=device)
        loaded_num_classes = checkpoint.get('num_classes', num_block_types) # Fallback to global
    except FileNotFoundError:
        print(f"Error: Checkpoint file {checkpoint_path} not found. Cannot run inference.")
        exit()
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}. Cannot run inference.")
        exit()

    # Initialize a fresh model for inference
    inference_model = Simple3DUNet(num_classes=loaded_num_classes).to(device)
    try:
        inference_model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}. Ensure model architecture matches checkpoint.")
        print(f"Attempting to load into the model used for training (if available and training was not commented out)... ")
        # Fallback: try loading into the model instance that might have been trained in the same run
        # This is less ideal if training was commented out.
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            inference_model = model # Use the already configured model
            print("Successfully loaded state_dict into the existing model instance.")
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}. Exiting.")
            exit()
    except KeyError as e:
        print(f"Error: Key '{e}' not found in checkpoint. Checkpoint structure might be incorrect.")
        exit()
        
    inference_model.eval() # Set model to evaluation mode

    loaded_class_to_block_id = checkpoint['class_to_block_id']
    
    print(f"Model loaded. Generating sample for prompt: '{prompt}'...")
    result = sample_categorical(prompt, inference_model, text_encoder, tokenizer, 
                              num_classes=loaded_num_classes, 
                              class_to_block_id=loaded_class_to_block_id, 
                              steps=num_timesteps) # Use full num_timesteps for sampling
    np.save(output_path, result)
    print(f"Generated voxel saved to {output_path}")

