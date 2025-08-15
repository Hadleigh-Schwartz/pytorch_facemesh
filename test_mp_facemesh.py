from mp_facemesh import PyTorchMediapipeFaceLandmarker
import cv2
import torch
import numpy as np
from utils import compare_to_real_mediapipe, compute_method_differences

def img_demo(img_path, output_name = None, display_annotation = True, display_landmark_comparison = False):
    """
    Test the PyTorch MedaPippe Face Mesh implementation on a single image and compare to the original MediaPipe implementation.
    In addition to visualizaing the landmarks on the input image and the blendshape scoers, this function demonstrates
    alignment of the landmarks with a differentiable version of the landmark alignment procedure Google uses.

    Parameters:
        img_path: str
            path to input image
        output_name: str
            if not None, save the output images with this name
        display_annotation: bool
            if True, display the landmarks and blendshape scores on the input image
        display_landmark_comparison: bool
            if True, display a comparison of the landmarks from the PyTorch and original MediaPipe implementations
    
    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp = PyTorchMediapipeFaceLandmarker(device, long_range_face_detect=False, short_range_face_detect=False).to(device)

    # validation on image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = True).to(device)
    landmarks, blendshapes, padded_face = mp(img_tensor)

    # vis and compare
    padded_face = padded_face.detach().cpu().numpy().astype(np.uint8)
    blendshapes_np = blendshapes.detach().cpu().numpy()
    landmarks_np = landmarks.detach().cpu().numpy()
    compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, output_name=output_name, display_annotation=display_annotation, display_landmark_comparison=display_landmark_comparison)
    landmarks_diff, blendshapes_diff = compute_method_differences(landmarks_np, blendshapes_np, padded_face)
    print(f"Avg. landmark pixel error: {landmarks_diff}")
    print(f"Blendshapes difference: {blendshapes_diff}")

def webcam_demo():
    """
    Test the PyTorch MediaPipe Face Mesh implementation on webcam input for live visualization.
    """
    # webcam live demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp = PyTorchMediapipeFaceLandmarker(device, short_range_face_detect=True).to(device)
    cap = cv2.VideoCapture(0) # 0 is typically webcam ID
    count = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        count += 1
        if count < 4:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img, dtype=torch.float32, requires_grad = False).to(device) # emulate format that will be output by generator
        landmarks, blendshapes, padded_face = mp(img_tensor)
        padded_face = padded_face.detach().numpy().astype(np.uint8)
        blendshapes_np = blendshapes.detach().numpy()
        compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, live_demo = True)

img_demo("data/harry.jpg", display_landmark_comparison=True, display_annotation=True, output_name="harry")
# webcam_demo()


