import cv2
import torch
from torch import nn
import numpy as np
from torchvision import transforms

from eye_landmarks_ids import eye_landmark_ids
from facenet_pytorch.models.mtcnn import MTCNN
from BlazeFace_PyTorch.blazeface import BlazeFace
from mediapipe_pytorch.facial_landmarks.facial_lm_model import FacialLM_Model 
from  mediapipe_pytorch.iris.irismodel import IrisLM
from deconstruct_mediapipe.blendshape_info import BLENDSHAPE_MODEL_LANDMARKS_SUBSET
from deconstruct_mediapipe.mlp_mixer import MediaPipeBlendshapesMLPMixer

def pad_image(im, desired_size=192):
    """
    Modified version of pad_image in mediapipe_pytorch/facial_landmarks/utils.py 
    to work with torch tensors

    The cv2.resize function is replaced with torchvision.transforms.Resize
        cv2.copyMakeBorder is replaced with transforms.Pad

    Parameters:
        im: torch.Tensor
            Image tensor of shape (H, W, 3) to be resized and padded
        desired_size: int
            The desired size of the output image, both width and height will be set to this value
    """
    old_size = im.shape[:2] # old_size is in (height, width) format

    if old_size[0] == 0 or old_size[1] == 0:
        return torch.zeros((desired_size, desired_size, 3), dtype=torch.float32)

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.permute(2, 0, 1)
    im = transforms.Resize(new_size)(im)
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # color = [0, 0, 0]
    # new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #     value=color)
    new_im = transforms.Pad([left, top, right, bottom])(im)
    new_im = new_im.permute(1, 2, 0)
    return new_im

class PyTorchMediapipeFaceLandmarker(nn.Module):
    def __init__(self, device = "cpu", long_range_face_detect = False, short_range_face_detect = False):
        """
        Attempt at a pure PyTorch re-implementation of MediaPipe Face Landmarker, including initial face detection and
        extraction of facial landmarks, iris landmarks, and blendshapes.

        Parameters:
            device: str
                The device to run the model on, either "cuda" or "cpu"
            long_range_face_detect: bool
                Whether to use MTCNN for long-range face detection for cropping. This is because BlazeFace is not as good at
                detecting faces that are far away, and MTCNN is generatlly more accurate. However, this is NOT DIFFERENTIABLE,
                so don't use if you are using this model in a pipeline that must be differentiable.
            short_range_face_detect: bool
                Whether to use BlazeFace for short-range face detection for cropping. This is differentiable and is the detector used under
                the hood by the actual MediaPipe model, but it doesn't have as good accuracy as MTCNN. It sometimes cuts off
                part of the face, which makes the predicted landmarks quite bad.
        """
        super(PyTorchMediapipeFaceLandmarker, self).__init__()
        
        self.long_range_face_detect = long_range_face_detect
        self.short_range_face_detect = short_range_face_detect

        if device == "cuda":
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        print("Initializing PyTorchMediapipeFaceLandmarker on device:", self.device)

        if self.long_range_face_detect:
            # face detection
            self.mtcnn = MTCNN(thresholds = [0.6, 0.6, 0.6], device = self.device) # more generous thresholds to ensure face is detected
        
        if self.short_range_face_detect:
            # blazeface face detection
            self.blaze_face = BlazeFace().to(self.device)
            self.blaze_face.load_weights("BlazeFace_PyTorch/blazeface.pth")
            self.blaze_face.load_anchors("BlazeFace_PyTorch/anchors.npy")
            back_net = BlazeFace(back_model=True).to(self.device)
            back_net.load_weights("BlazeFace_PyTorch/blazefaceback.pth")
            back_net.load_anchors("BlazeFace_PyTorch/anchorsback.npy")
            self.blaze_face.min_score_thresh = 0.4
            self.blaze_face.min_suppression_threshold = 0.3
            
        # facial landmarks
        self.facelandmarker = FacialLM_Model().to(self.device)
        self.facelandmarker_weights = torch.load("mediapipe_pytorch/facial_landmarks/model_weights/facial_landmarks.pth")
        self.facelandmarker.load_state_dict(self.facelandmarker_weights)
        self.facelandmarker = self.facelandmarker.eval() 

        # iris landmarks
        self.irislandmarker = IrisLM().to(self.device)
        self.irislandmarker_weights = torch.load("mediapipe_pytorch/iris/model_weights/irislandmarks.pth")
        self.irislandmarker.load_state_dict(self.irislandmarker_weights)
        self.irislandmarker = self.irislandmarker.eval() 

        # blendshapes from facial and iris landmarks
        self.blendshape_model = MediaPipeBlendshapesMLPMixer().to(self.device)
        self.blendshape_model.load_state_dict(torch.load("deconstruct_mediapipe/face_blendshapes.pth"))

    def detect_face(self, img, padding = 50):
        """
        Detects face in image using MTCNN and returns cropped face tensor

        WARNING: This is not differentiable, it's just here for
        the purpose of checking if a face was detected.

        Parameters:
            img: torch.Tensor H x W x 3
                The image to detect the face in
            padding: int
                The amount of padding to add around the detected face
        
        Returns:
            torch.Tensor
                Image cropped to detected face
        """
        bboxes, probs = self.mtcnn.detect(img)
        if bboxes is None: # if no face detected
            return torch.ones(img.shape, dtype=torch.float32)*-1 # placeholder for Nans
        bbox = bboxes[0]
        bbox = bbox + [-padding, -padding, padding, padding] # add padding to the bbox, based on observation that mediapipe landmarks extractor benefits from this
        x1, y1, x2, y2 = bbox.astype(int)
        cropped_face_tensor = img[y1:y2, x1:x2, :]
        return cropped_face_tensor
    

    def preprocess_face_for_landmark_detection(self, img):
        """
        Apply preprocessing steps required by mediapipe_pytorch facial landmarker model. 
        See example usage in mediapipe_pytorch/facial_landmarks/inference.py to understand why this is required.

        Parameters:
            img: torch.Tensor
                Image of face to preprocess
        
        Returns:
            torch.Tensor
                Preprocessed face image
        """
        proc_img = pad_image(img, desired_size=192) # resize/pad
        proc_img = (proc_img / 127.5) - 1.0 # normalize 
        proc_img = proc_img.permute(2, 0, 1) # per line 116 of mediapipe_pytorch/facial_landmarks/inference.py, blob is expected to have dimensions [3, H, W]
        return proc_img

    def unnormalize_face(self, img):
        """
        Undo the normalization and permutation applied for mediapipe-pytorch facial landmark detection, but leave the padding it added. 
        This is for obtaining an image that can be displayed for visualization purposes.

        Parameters:
            img: torch.Tensor
                Image of face, previously prepreocessed using preprocess_face_for_landmark_detection
        
        Returns:
            torch.Tensor
                Unnormalized face image
        """
        img = img.permute(1, 2, 0)
        img = (img + 1.0) * 127.5
        return img

    def preprocess_eye_for_landmark_detection(self, img):
        """
        Apply preprocessing steps required by mediapipe_pytorch iris landmarker model.
        See example usage in mediapipe_pytorch/iris/inference.py to understand why this is required.

        Parameters:
            img: torch.Tensor
                Image of eye to preprocess
        
        Returns:
            torch.Tensor
                Preprocessed eye image
        """
        # preprocess img
        proc_img = pad_image(img, desired_size=64) # resize
        # check if face is all zeros
        if torch.all(proc_img == 0):
            return proc_img
        proc_img /= 255 # normalize
        proc_img = proc_img.permute(2, 0, 1) # per line 116 of mediapipe_pytorch/facial_landmarks/inference.py, blob is expected to have dimensions [3, H, W]
        return proc_img

    def unnormalize_eye(self, img):
        """
        Undo the normalization and permutation applied for mediapipe-pytorch iris landmark detection, but leave the padding it added.
        This is for obtaining an image that can be displayed for visualization purposes.

        Parameters:
            img: torch.Tensor
                Image of eye, previously preprocessed using preprocess_eye_for_landmark_detection

        Returns:
            torch.Tensor
                Unnormalized eye image
        """
        img *= 255
        img = img.permute(1, 2, 0)
        return img


    def get_eye_bounds(self, facial_landmarks, eye = "left"):
        """
        Compute boundaries of the eyes based on the facial landmarks.

        Based off of this resource (https://github.com/Morris88826/MediaPipe_Iris/blob/main/README.md), the iris model expects a 64x64 image of the eye
        cropped according to the  (rightEyeUpper0, rightEyeLower0, leftEyeUpper0, leftEyeLower0) landmarks, whose indices are specified 
        here: https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
        
        This resource from Google https://github.com/google-ai-edge/mediapipe/blob/ab1de4fced96c18b79eda4ab407b04eb08301ea4/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt
        seems to only consider 33 and 133 as denoting the boundaries of the left eye, and 263 and 362 as the right. Note that it seems to flip 
        left/right relative to the above resource on landmark indices (https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts)
        Then it adds a 25% margin to each side and it appears to set the height==width. Below I use this approach.

        Parameters:
            facial_landmarks: torch.Tensor
                Tensor of shape (468, 3) containing the facial landmarks, where each landmark is a 3D point (x, y, z)
            eye: str
                Either "left" or "right", specifying which eye to compute the bounds for
        
        Returns:
            tuple: (eye_left, eye_right, eye_top, eye_bottom, eye_width, eye_height)
                The left and right x-coordinates, top and bottom y-coordinates, width and height of the eye crop.
                The width and height are equal, as per the MediaPipe implementation.
                The coordinates are in the same pixel space as the facial landmarks.
                The eye crop is centered around the eye, with a 25% margin added to the left and right sides.
                The top and bottom coordinates are adjusted to ensure the crop is square.
        """
        right_eye_ids = [362, 263]
        left_eye_ids = [33, 133]
        if eye == "right":
            eye_left = facial_landmarks[right_eye_ids[0], 0].item()
            eye_right = facial_landmarks[right_eye_ids[1], 0].item()
            eye_top = facial_landmarks[right_eye_ids, 1].min().item()
            eye_bottom = facial_landmarks[right_eye_ids, 1].max().item()    
        if eye == "left":
            eye_left = facial_landmarks[left_eye_ids[0], 0].item()
            eye_right = facial_landmarks[left_eye_ids[1], 0].item()
            eye_top = facial_landmarks[left_eye_ids, 1].min().item()
            eye_bottom = facial_landmarks[left_eye_ids, 1].max().item()
        eye_width = eye_right - eye_left
        horizontal_margin = 0.25 * eye_width 
        eye_left -= horizontal_margin
        eye_right += horizontal_margin
        eye_top -= eye_width // 2
        eye_top -= horizontal_margin
        eye_bottom += eye_width // 2
        eye_bottom += horizontal_margin
        eye_width = eye_width + 2*horizontal_margin
        eye_height = eye_width
            
        return eye_left, eye_right, eye_top, eye_bottom, eye_width, eye_height

    def vis_facial_landmarks(self, img, facial_landmarks):
        """
        Visualize the landmarks on an image. For debugging purposes.

        Parameters:
            img: torch.Tensor
                Image tensor of shape H x W x 3, RGB
            facial_landmarks: torch.Tensor
                Tensor of shape (468, 3) containing the facial landmarks, where each landmark is a 3D point (x, y, z) or 2D point (x, y) if z is not used.
                The landmarks are in the same pixel space as the image.
        
        Returns:
            None, but displays the image with landmarks drawn on it.
        """
        img = img.detach().cpu().numpy().astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(facial_landmarks.shape[0]):
            coord = facial_landmarks[i, :]
            x, y = coord[0].item(), coord[1].item()
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.imshow("Landmarks", img)
        cv2.waitKey(0)

    def vis_eye_cropping(self, img, right_eye_bounds, left_eye_bounds):
        """
        Visualize the eye cropping on the image. For debugging purposes.

        Parameters:
            img: torch.Tensor
                Image tensor of shape H x W x 3, RGB
            right_eye_bounds: tuple
                Tuple of (left, right, top, bottom) coordinates for the right eye crop
            left_eye_bounds: tuple
                Tuple of (left, right, top, bottom) coordinates for the left eye crop
        
        Returns:
            None, but displays the image with rectangles drawn around the eye crops.
        """
        img = img.deatch().cpu().numpy().astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        right_eye_left, right_eye_right, right_eye_top, right_eye_bottom = right_eye_bounds
        left_eye_left, left_eye_right, left_eye_top, left_eye_bottom = left_eye_bounds

        cv2.rectangle(img, (int(right_eye_left), int(right_eye_top)), (int(right_eye_right), int(right_eye_bottom)), (0, 255, 0), 2)
        cv2.rectangle(img, (int(left_eye_left), int(left_eye_top)), (int(left_eye_right), int(left_eye_bottom)), (0, 255, 0), 2)

        cv2.imshow("Eye Cropping", img)
        cv2.waitKey(0)

    def iris_landmarks_to_original_pixel_space(self, iris_landmarks, eye_bounds, eye_crop_dims):
        """
        The iris landmarks were computed on a 64x64 padded image of the eye crop, obtained by extracting a crop from the 192x192
        and then resizing in pad_image.
        
        We need to convert these eye-crop-centeric landmarks to be in same pixel space that the facial landmarks were computed in, 
        so we first unscale the landmarks back to the 192x192 space, then adjust them according to the boundaries of the eye crop
        w.r.t to the 192x192 image. 

        Parameters:
            iris_landmarks: torch.Tensor
                Tensor of shape (5, 3) containing the iris landmarks, where each landmark is a 3D point (x, y, z)
            eye_bounds: tuple
                Tuple of (left, right, top, bottom) coordinates for the eye crop
            eye_crop_dims: tuple
                Tuple of (width, height) for the eye crop dimensions
        
        Returns:
            torch.Tensor
                Tensor of shape (5, 3) containing the iris landmarks in the original facial pixel space
        """
        left, right, top, bottom = eye_bounds
        eye_width, eye_height = eye_crop_dims
        ratio = float(64)/max(eye_height, eye_width) #why *2
        new_eye_size = tuple([int(x*ratio) for x in [eye_height, eye_width]])
        eye_delta_w = 64 - new_eye_size[1]
        eye_delta_h = 64 - new_eye_size[0]
        eye_top_padding, eye_bottom_padding = eye_delta_h//2, eye_delta_h-(eye_delta_h//2)
        eye_left_padding, eye_right_padding = eye_delta_w//2, eye_delta_w-(eye_delta_w//2)

        iris_landmarks[:, 0] = iris_landmarks[:, 0] - eye_left_padding
        iris_landmarks[:, 0] = iris_landmarks[:, 0]*(eye_width/new_eye_size[1]) + left
        iris_landmarks[:, 1] = iris_landmarks[:, 1] - eye_top_padding
        iris_landmarks[:, 1] = iris_landmarks[:, 1]*(eye_height/new_eye_size[0]) + top
        return iris_landmarks


    def forward(self, img_tensor):
        """
        Parameters:
            img_tensor: torch.Tensor
                Image tensor, H x W x 3, RGB. Should be a crop of a face.

        Returns:
            landmarks: torch.Tensor
                478 x 3 tensor of FaceMesh dense landmarks. 
            blendshapes: torch.Tensor
                52 tensor of blendshape scores
        """

        if self.long_range_face_detect:
            long_range_cropped_face = self.detect_face(img_tensor)
            if torch.all(long_range_cropped_face == 0) or \
                ((long_range_cropped_face.shape[0] / long_range_cropped_face.shape[1]) > 3) or \
                long_range_cropped_face.shape[0] < 100 or long_range_cropped_face.shape[1] < 100:
                landmarks_zeroes = torch.zeros(478, 3)
                blendshapes_zeroes = torch.zeros(52)
                face_zeroes = torch.zeros(img_tensor.shape)
                return landmarks_zeroes, blendshapes_zeroes, face_zeroes
            else:
                img_tensor = long_range_cropped_face
        
        if self.short_range_face_detect:
            short_range_cropped_face = self.blaze_face.predict_on_image(img_tensor)
            if torch.all(short_range_cropped_face == 0) or \
                ((short_range_cropped_face.shape[0] / short_range_cropped_face.shape[1]) > 3) or \
                short_range_cropped_face.shape[0] < 100 or short_range_cropped_face.shape[1] < 100:
                landmarks_zeroes = torch.zeros(478, 3)
                blendshapes_zeroes = torch.zeros(52)
                face_zeroes = torch.zeros(img_tensor.shape)
                return landmarks_zeroes, blendshapes_zeroes, face_zeroes
            else:
                img_tensor = short_range_cropped_face

        proc_face = self.preprocess_face_for_landmark_detection(img_tensor)

        # run facial landmark detection==
        facial_landmarks, confidence = self.facelandmarker.predict(proc_face) # predict
        facial_landmarks = facial_landmarks[0, :, :, :] # assume there is only one face in the image, so take the first set of landmarks
        facial_landmarks = facial_landmarks.view(468, 3)
        
        # only for visualization
        padded_face = self.unnormalize_face(proc_face) # undo normalization and permutation applied for facial landmark detection, but leave the padding it added
        
        # self.vis_facial_landmarks(padded_face, facial_landmarks) # debugging only

        # get eye bounds
        right_eye_left, right_eye_right, right_eye_top, right_eye_bottom, right_eye_width, right_eye_height = self.get_eye_bounds(facial_landmarks, eye = "right")
        left_eye_left, left_eye_right, left_eye_top, left_eye_bottom, left_eye_width, left_eye_height = self.get_eye_bounds(facial_landmarks, eye = "left")
        
        if right_eye_width == 0 or right_eye_height == 0 or left_eye_width == 0 or left_eye_height == 0:
            landmarks_zeroes = torch.zeros(478, 3)
            blendshapes_zeroes = torch.zeros(52)
            face_zeroes = torch.zeros(img_tensor.shape)
            return landmarks_zeroes, blendshapes_zeroes, face_zeroes

        padded_face_H, padded_face_W = padded_face.shape[:2]
        left_eye_crop = padded_face[max(0, int(left_eye_top)):min(int(left_eye_bottom), padded_face_H), max(0, int(left_eye_left)):min(int(left_eye_right), padded_face_W), :]
        right_eye_crop = padded_face[max(0, int(right_eye_top)):min(int(right_eye_bottom), padded_face_H), max(0, int(right_eye_left)):min(int(right_eye_right), padded_face_W), :]

    
        # need to flip horizontally for right eye, according to https://github.com/Morris88826/MediaPipe_Iris, but I'm not sure
        # this is helping performance for us so I've made toggleable.
        flip = False
        if flip:
            right_eye_crop = torch.flip(right_eye_crop, [1])

        # debugging only
        # vis_eye_cropping(padded_face, facial_landmarks, (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom), (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom))
 
        # pad the eye crops
        proc_left_eye_crop = self.preprocess_eye_for_landmark_detection(left_eye_crop)    
        proc_right_eye_crop = self.preprocess_eye_for_landmark_detection(right_eye_crop)
        # check if either crop is all zeros
        if torch.all(proc_left_eye_crop == 0) or torch.all(proc_right_eye_crop == 0):
            landmarks_zeroes = torch.zeros(478, 3)
            blendshapes_zeroes = torch.zeros(52)
            face_zeroes = torch.zeros(img_tensor.shape)
            return landmarks_zeroes, blendshapes_zeroes, face_zeroes

        # run iris detection
        left_eye_contour_landmarks, left_iris_landmarks = self.irislandmarker.predict(proc_left_eye_crop)
        right_eye_contour_landmarks, right_iris_landmarks = self.irislandmarker.predict(proc_right_eye_crop)
        left_iris_landmarks = left_iris_landmarks.view(5, 3)
        right_iris_landmarks = right_iris_landmarks.view(5, 3)
        left_eye_contour_landmarks = left_eye_contour_landmarks.view(71, 3)
        right_eye_contour_landmarks = right_eye_contour_landmarks.view(71, 3)
        if flip:
            # experimental
            right_iris_landmarks_flipped = right_iris_landmarks.clone()
            right_iris_landmarks_flipped[:, 0] = 64 - right_iris_landmarks_flipped[:, 0]
            right_iris_landmarks = right_iris_landmarks_flipped
            right_eye_contour_landmarks_flipped = right_eye_contour_landmarks.clone()
            right_eye_contour_landmarks_flipped[:, 0] = 64 - right_eye_contour_landmarks_flipped[:, 0]
            right_eye_contour_landmarks = right_eye_contour_landmarks_flipped
        
        """
        refine the eye contour landmarks outputted by the 468-point model based on output of the iris landmarks
        Per here https://github.com/google-ai-edge/mediapipe/blob/ab1de4fced96c18b79eda4ab407b04eb08301ea4/mediapipe/graphs/iris_tracking/calculators/update_face_landmarks_calculator.cc#L33
        and here https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md 
        this is how the eventual eye-related 478-landmarkers in MP implementation are obtained.
        """
        # eye landmarks to refine based on iris landmarker output, in case you want to only replace some
        keep_eye_ids = [i for i in range(71)]#[i for i in range(0, 48)] + [i for i in range(54,63)] # dont use eyebrows stuff
        
        # debugging only
        # all_left_eye_landmarks = torch.cat([left_eye_contour_landmarks[keep_eye_ids], left_iris_landmarks], dim=0)
        # all_right_eye_landmarks = torch.cat([right_eye_contour_landmarks[keep_eye_ids], right_iris_landmarks], dim=0)
        # padded_left_eye = self.unnormalize_eye(proc_left_eye_crop)
        # padded_right_eye = self.unnormalize_eye(proc_right_eye_crop)
        # self.vis_facial_landmarks(padded_left_eye, all_left_eye_landmarks) 
        # self.vis_facial_landmarks(padded_right_eye, all_right_eye_landmarks)

        # adjust the iris landmarks to the original image pixel space
        left_iris_landmarks = self.iris_landmarks_to_original_pixel_space(left_iris_landmarks, (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom), (left_eye_width, left_eye_height))
        right_iris_landmarks = self.iris_landmarks_to_original_pixel_space(right_iris_landmarks, (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom), (right_eye_width, right_eye_height))
        
        # replace the original facial landmarker eye contour landmarks with the refined ones from iris landmarker
        left_eye_contour_landmarks = left_eye_contour_landmarks[keep_eye_ids, :]
        right_eye_contour_landmarks = right_eye_contour_landmarks[keep_eye_ids, :]
        refined_left_eye_landmarks = self.iris_landmarks_to_original_pixel_space(left_eye_contour_landmarks, (left_eye_left, left_eye_right, left_eye_top, left_eye_bottom), (left_eye_width, left_eye_height))
        refined_right_eye_landmarks = self.iris_landmarks_to_original_pixel_space(right_eye_contour_landmarks, (right_eye_left, right_eye_right, right_eye_top, right_eye_bottom), (right_eye_width, right_eye_height))
        facial_landmarks[eye_landmark_ids, :] = torch.cat([refined_left_eye_landmarks, refined_right_eye_landmarks], dim=0)

        # debugging only
        # self.vis_facial_landmarks(padded_face, left_iris_landmarks)
        # self.vis_facial_landmarks(padded_face, right_iris_landmarks)

        """
        append the iris landmarks to the general landmarks array in the proper order
        know from https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
        [468, 469, 470, 471, 472] are left iris landmarks
        Confirmed by analyzing and comparing to here https://github.com/k-m-irfan/simplified_mediapipe_face_landmarks
        that the indices of the left iris landmarks are 468, 469, 470, 471, 472, right iris landmarks are 473, 474, 475, 476, 477
        (i.e., ordered same as in the MediaPipe 478-landmarker model), therefore can just append in order, left first, then right
        """
        all_landmarks = torch.cat([facial_landmarks, left_iris_landmarks, right_iris_landmarks], dim=0)
        
        # The z coordinate of the landmarks is scaled by the height of the image, unlike in the case of the actual MediaPipe model.
        # To convert to the same scale as the actual MediaPipe model, we need to divide the z coordinate by the height of the image.
        all_landmarks[:, 2] = all_landmarks[:, 2] / padded_face.shape[0]

        # unlike in case of landmarks outputted by the actual mediapipe model, ours currently are already
        # scaled by the image dimensions, so no need to perform line 96 in deconstruct-mediapipe/test_converted_model.py
        blendshape_input = all_landmarks[BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2]
        blendshape_input = blendshape_input.unsqueeze(0) # add batch dimension

        blendshapes = self.blendshape_model(blendshape_input)
        blendshapes = blendshapes.squeeze()

        # clip blendshapes to be between 0 and 1 because the real MP does not predict values outside this range
        blendshapes = torch.clamp(blendshapes, 0, 1) 

        return all_landmarks, blendshapes, padded_face


