#testing
import cv2
import torch
import torch.nn as nn
import numpy as np

# --- 1. Define the Model Architecture (Must match training exactly) ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(16 + 32, 16)
        self.conv_last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.dconv_down3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        return torch.sigmoid(self.conv_last(x))

# --- 2. Inference Function ---
def run_tongue_tracking(video_path, model_path):
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode (freezes dropout/batchnorm)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get original dimensions for resizing back later
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # A. Preprocessing (Match Training Logic)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to 128x128 for the AI
        input_tensor = cv2.resize(gray, (128, 128))
        input_tensor = input_tensor.astype(np.float32) / 255.0 # Normalize
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 128, 128]
        input_tensor = input_tensor.to(device)

        # B. Inference
        with torch.no_grad():
            prediction = model(input_tensor)

        # C. Post-processing
        # Remove batch dims -> [128, 128]
        pred_mask = prediction.squeeze().cpu().numpy()

        # Threshold (Prob > 0.5 is Tongue)
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

        # Resize mask BACK to original video size
        # Use NEAREST neighbor to keep sharp edges (0 or 1), not fuzzy interpolation
        full_size_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # D. Contour Drawing
        contours, _ = cv2.findContours(full_size_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw green contours on original frame
        # (0, 255, 0) = Green color, 2 = thickness
        if contours:
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # E. Display
        cv2.imshow('Tongue Tracking (U-Net)', frame)

        # Optional: Show the raw mask in a separate window to debug
        # cv2.imshow('AI Mask', full_size_mask * 255)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 3. Run It ---
if __name__ == "__main__":
    # Replace with your actual paths
    TEST_VIDEO = r"ultrasound_videos\test\1212\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip\1212_F_G1_050.mp4"
    MODEL_PATH = "largest_blob.pth"

    run_tongue_tracking(TEST_VIDEO, MODEL_PATH)

    #ultrasound_videos\train\1121\DAT_DICOM\Midsaggital_Videos\Horizontal_Flip\1121_M_G1_003.mp4

    """
    python test.py
    python model.py
    """