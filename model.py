import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten the tensors to 1D vectors
        # inputs: (Batch, Channel, Height, Width) -> (Batch * C * H * W)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Intersection (A ∩ B)
        intersection = (inputs * targets).sum()
        
        # Calculate Union (|A| + |B|)
        total = inputs.sum() + targets.sum()
        
        # Dice Coefficient = (2 * Intersection + smooth) / (Total + smooth)
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        
        # Return 1 - Dice (because we want to MINIMIZE loss)
        return 1 - dice

# --- PART 1: The Modified Gabor Generator (Your "Teacher") ---
def generate_gabor_ground_truth(frame):
    """
    Runs your exact Gabor logic on a single frame to create the training label.
    Returns: Binary Mask (0 or 1)
    """
    # Pre-processing
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    denoised = cv2.medianBlur(gray, 19)

    # Gabor Filter Bank
    accumulated_response = np.zeros_like(denoised, dtype=np.float32)
    ksize, sigma, lambd, gamma = 31, 3.0, 8.0, 0.5
    
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(denoised, cv2.CV_32F, kern)
        np.maximum(accumulated_response, fimg, accumulated_response)

    # Thresholding & Masking
    threshold_value = np.percentile(accumulated_response, 88)
    accumulated_response[accumulated_response < threshold_value] = 0
    gabor_8u = cv2.normalize(accumulated_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(gabor_8u, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    
    # Return normalized binary mask (0.0 to 1.0) for the Neural Net
    return (closed > 0).astype(np.float32)

# --- PART 2: The Custom Dataset (First Frame Only) ---
class UltrasoundFirstFrameDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        Args:
            video_dir (str): Path to folder containing .mp4 or .avi files
        """
        self.video_paths = glob.glob(os.path.join(video_dir, "*.mp4")) # Adjust extension as needed
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # 1. Open Video
        cap = cv2.VideoCapture(self.video_paths[idx])
        ret, frame = cap.read() # Read ONLY Frame 0
        cap.release()
        
        if not ret:
            # Handle bad video (return zeros or skip)
            return torch.zeros((1, 128, 128)), torch.zeros((1, 128, 128))

        # 2. Generate Ground Truth using the "Teacher" function
        mask = generate_gabor_ground_truth(frame)
        
        # 3. Prepare Input (Raw Image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 4. Resize to standard U-Net size (e.g., 128x128 or 256x256)
        # Neural nets need consistent dimensions
        gray_resized = cv2.resize(gray, (128, 128))
        mask_resized = cv2.resize(mask, (128, 128))
        
        # 5. Convert to PyTorch Tensors (C, H, W)
        image_tensor = torch.from_numpy(gray_resized).float() / 255.0 # Normalize 0-1
        mask_tensor = torch.from_numpy(mask_resized).float()
        
        return image_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)

# --- PART 3: The U-Net Model (The "Student") ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Helper for Double Convolution layers
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling)
        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        
        self.maxpool = nn.MaxPool2d(2)
        
        # Decoder (Upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(16 + 32, 16)
        
        self.conv_last = nn.Conv2d(16, 1, 1) # Output: 1 channel (probability map)
        
    def forward(self, x):
        # Down 1
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        # Down 2
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        # Bridge (Down 3)
        x = self.dconv_down3(x)
        
        # Up 1
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1) # Skip Connection
        x = self.dconv_up2(x)
        
        # Up 2
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1) # Skip Connection
        x = self.dconv_up1(x)
        
        # Final output (Sigmoid to get 0-1 probability)
        return torch.sigmoid(self.conv_last(x))


# --- PART 4: Training Loop ---
if __name__ == "__main__":
    # 1. Setup Device
    # Check if CUDA is actually available, otherwise fallback (or crash if you strictly need CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Data
    video_folder = "/content/drive/MyDrive/Research_models/ultrasound_videos/train/0123/DAT_DICOM/Midsaggital_Videos/Horizontal_Flip" 
    dataset = UltrasoundFirstFrameDataset(video_folder)
    
    # num_workers=2 helps load data in parallel on CPU while GPU trains
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # 3. Setup Model & Loss
    model = SimpleUNet().to(device)  # <--- CRITICAL: Move model to GPU
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DiceLoss() 

    print(f"Training on {len(dataset)} videos (Frame 0 only)...")

    for epoch in range(50):
        epoch_loss = 0
        model.train() # Set model to training mode
        
        for images, masks in dataloader:
            # images shape: [batch, 1, 128, 128]
            # masks shape:  [batch, 1, 128, 128]
            
            # 4. Move Data to GPU
            images = images.to(device) # <--- CRITICAL
            masks = masks.to(device)   # <--- CRITICAL
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")

    print("Training Complete. Model saved.")
    torch.save(model.state_dict(), "tongue_tracker_unet.pth")