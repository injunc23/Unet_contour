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
    
def get_files_from_folders(folder_list, extension="*.mp4"):
    """
    Given a list of folder paths, returns a single list of all video files inside them.
    """
    all_files = []
    for folder in folder_list:
        # Get all files with the specific extension in this folder
        files = glob.glob(os.path.join(folder, extension))
        if not files:
            print(f"Warning: No files found in {folder}")
        all_files.extend(files)
    return all_files

# --- PART 1: The Modified Gabor Generator (Your "Teacher") ---
"""
Traditional gabor is making model trace too much noise and the gradient scale on the left of video.
"""
def generate_gabor_ground_truth(frame):#roi based gabor using gaussian spotlight
    # 1. Standard Pre-processing
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    denoised = cv2.medianBlur(gray, 19)
    rows, cols = denoised.shape

    # --- NEW: GENERATE SOFT ROI (THE SPOTLIGHT) ---
    # Create a 2D Gaussian distribution centered in the image
    # sigma_x and sigma_y control the "width" of the spotlight
    # Larger sigma = wider spotlight. Smaller sigma = tighter focus.
    sigma_x = cols * 0.5  # Spotlight covers ~50% of the width clearly
    sigma_y = rows * 0.5 
    
    X = cv2.getGaussianKernel(cols, sigma_x)
    Y = cv2.getGaussianKernel(rows, sigma_y)
    gaussian_map = Y * X.T  # Matrix multiplication to create 2D map
    
    # Normalize the spotlight so the center is exactly 1.0
    gaussian_map = gaussian_map / np.max(gaussian_map)
    
    # Optional: Shift the spotlight down slightly? 
    # (Since tongues are usually in the bottom half). 
    # If you strictly want NO hard-coding, keep it centered.
    # To shift without hard numbers, you can roll the array:
    # gaussian_map = np.roll(gaussian_map, int(rows * 0.1), axis=0) # Shifts down 10%
    
    # ------------------------------------------------

    # 2. Gabor Filter Bank (Existing code)
    accumulated_response = np.zeros_like(denoised, dtype=np.float32)
    ksize, sigma, lambd, gamma = 31, 3.0, 8.0, 0.5
    
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(denoised, cv2.CV_32F, kern)
        np.maximum(accumulated_response, fimg, accumulated_response)

    # 3. APPLY THE SPOTLIGHT
    # This dampens noise at the edges/top/bottom gently
    weighted_response = accumulated_response * gaussian_map

    # 4. Thresholding & Masking (Run on the WEIGHTED response)
    # The percentile threshold will now naturally pick the center features
    # because the edge noise has been "dimmed" by the spotlight.
    threshold_value = np.percentile(weighted_response, 92) # Increased slightly to 92
    weighted_response[weighted_response < threshold_value] = 0
    
    gabor_8u = cv2.normalize(weighted_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 5. Morphological Cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(gabor_8u, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    
    return (closed > 0).astype(np.float32)

# def generate_gabor_ground_truth(frame): #this is the gabor tracking the largest blob
#     # ... (Your existing Gabor + Thresholding code here) ...
#     if len(frame.shape) == 3:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = frame

#     denoised = cv2.medianBlur(gray, 19)

#     # Gabor Filter Bank
#     accumulated_response = np.zeros_like(denoised, dtype=np.float32)
#     ksize, sigma, lambd, gamma = 31, 3.0, 8.0, 0.5

#     for theta in np.arange(0, np.pi, np.pi / 8):
#         kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
#         fimg = cv2.filter2D(denoised, cv2.CV_32F, kern)
#         np.maximum(accumulated_response, fimg, accumulated_response)

#     # Thresholding & Masking
#     threshold_value = np.percentile(accumulated_response, 90)
#     accumulated_response[accumulated_response < threshold_value] = 0
#     gabor_8u = cv2.normalize(accumulated_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     # Morphological cleaning
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#     opened = cv2.morphologyEx(gabor_8u, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
#     # ... (After your Morphological Closing) ...
#     # closed = cv2.morphologyEx(...)
    
#     # --- NEW CLEANING STEP ---
#     # Find all connected blobs
#     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         return np.zeros_like(closed, dtype=np.float32)
    
#     # Identify the largest blob (The Tongue)
#     largest_blob = max(contours, key=cv2.contourArea)
    
#     # Create a clean black mask
#     clean_mask = np.zeros_like(closed)
    
#     # Draw ONLY the largest blob onto the clean mask
#     cv2.drawContours(clean_mask, [largest_blob], -1, 255, thickness=cv2.FILLED)
    
#     return (clean_mask > 0).astype(np.float32)

# def generate_gabor_ground_truth(frame): #this is the traditional gabor
#     """
#     Runs your exact Gabor logic on a single frame to create the training label.
#     Returns: Binary Mask (0 or 1)
#     """
#     # Pre-processing
#     if len(frame.shape) == 3:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = frame

#     denoised = cv2.medianBlur(gray, 19)

#     # Gabor Filter Bank
#     accumulated_response = np.zeros_like(denoised, dtype=np.float32)
#     ksize, sigma, lambd, gamma = 31, 3.0, 8.0, 0.5

#     for theta in np.arange(0, np.pi, np.pi / 8):
#         kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
#         fimg = cv2.filter2D(denoised, cv2.CV_32F, kern)
#         np.maximum(accumulated_response, fimg, accumulated_response)

#     # Thresholding & Masking
#     threshold_value = np.percentile(accumulated_response, 90)
#     accumulated_response[accumulated_response < threshold_value] = 0
#     gabor_8u = cv2.normalize(accumulated_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     # Morphological cleaning
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#     opened = cv2.morphologyEx(gabor_8u, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))

#     # Return normalized binary mask (0.0 to 1.0) for the Neural Net
#     return (closed > 0).astype(np.float32)

# --- PART 2: The Custom Dataset (First Frame Only) ---
class UltrasoundFirstFrameDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """
        Args:
            file_list (list): A list of full file paths to videos.
        """
        self.video_paths = file_list  # <--- CHANGED: Direct assignment
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # (This part remains exactly the same as before)
        cap = cv2.VideoCapture(self.video_paths[idx])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return torch.zeros((1, 128, 128)), torch.zeros((1, 128, 128))

        mask = generate_gabor_ground_truth(frame)
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        gray_resized = cv2.resize(gray, (128, 128))
        mask_resized = cv2.resize(mask, (128, 128))
        
        image_tensor = torch.from_numpy(gray_resized).float() / 255.0
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
    # --- A. Setup Paths ---
    # Define your specific folders here
    train_folders = [
        r"ultrasound_videos\train\0123\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\0130\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\0213\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\0305\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\0910\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\0919\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\1121\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\1125\DAT_DICOM\Midsaggital_Videos\No_Horizontal_Flip",
        r"ultrasound_videos\train\1107\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\1113\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip"
        
        # ... add all 10 folders
    ]
    
    val_folders = [
        r"ultrasound_videos\train\1017\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip",
        r"ultrasound_videos\train\1031\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip"
    ]
    
    # 1. Gather all files
    print("Gathering files...")
    train_files = get_files_from_folders(train_folders)
    val_files = get_files_from_folders(val_folders)
    
    print(f"Found {len(train_files)} training videos and {len(val_files)} validation videos.")

    # --- B. Setup DataLoaders ---
    # Create two separate datasets
    train_dataset = UltrasoundFirstFrameDataset(train_files)
    val_dataset = UltrasoundFirstFrameDataset(val_files)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    # Note: Shuffle=False for validation is standard (consistency)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # --- C. Setup Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DiceLoss() 

    # Track best performance to save only the best model
    best_val_loss = float('inf')

    # --- D. The Training Loop ---
    epochs = 50
    for epoch in range(epochs):
        
        # 1. TRAINING PHASE
        model.train() # Enable dropout/batchnorm
        train_loss = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # 2. VALIDATION PHASE
        model.eval() # Disable dropout/batchnorm
        val_loss = 0
        
        with torch.no_grad(): # Disable gradient calculation (saves RAM, faster)
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # 3. REPORTING
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 4. CHECKPOINTING (Save only if validation improves)
        if avg_val_loss < best_val_loss:
            print(f"   >>> Validation Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "gaussian_spotlight_roi.pth")

    print("Training Complete.")