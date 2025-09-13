
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101



def deeplabv3_predictor():
  # Load model and set to evaluation mode
  model = deeplabv3_resnet101(pretrained=True).eval()

  # Transform the image/frame for model input
  preprocess = T.Compose([
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  return model, preprocess