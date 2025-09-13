import sys
import argparse
import cv2
from .lib.preprocess import h36m_coco_format, revise_kpts
from .lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy

from .common.model_poseformer import PoseTransformerV2 as Model
from .common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from .visualize import show2Dpose, show3Dpose, showimage

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#--------------------------------------------------------------------------------------
def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successful!')

    os.makedirs(output_dir, exist_ok=True)
    output_npz = output_dir + '2d_keypoints_hrnet.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)
    output_npz = output_dir + '2d_keypoints_scores_hrnet.npz'
    np.savez_compressed(output_npz, reconstruction=scores)

    return keypoints, scores

#--------------------------------------------------------------------------------------
def get_pose3D(video_path, output_dir, keypoints, scores):
    num_frames = 243
    pad = (num_frames - 1) // 2

    ## Reload 
    model = nn.DataParallel(Model()).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of PoseFormerV2 in 'checkpoint/']
    model_path = sorted(glob.glob(os.path.join('build_dataset/PoseFormerV2/checkpoint/', '27_243_45.2.bin')))[0]

    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model_pos'], strict=True)

    model.eval()    

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    keypoints_3d = []
    print('\nGenerating 3D pose...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        ## input frames
        start = max(0, i - pad)
        end =  min(i + pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != num_frames:
            if i < pad:
                left_pad = pad - i
            if i > len(keypoints[0]) - pad - 1:
                right_pad = i + pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        # input_2D_no += np.random.normal(loc=0.0, scale=5, size=input_2D_no.shape)
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        # (2, 243, 17, 2)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## estimation
        output_3D_non_flip = model(input_2D[:, 0]) 
        output_3D_flip     = model(input_2D[:, 1])
        # [1, 1, 17, 3]

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        keypoints_3d.append(post_out)
        
        input_2D_no = input_2D_no[pad]

        ## 2D
        score_per_frame = scores[0][i]
        conf_joints_ids = list(np.where(score_per_frame > 0.3)[0])
        image = show2Dpose(input_2D_no, copy.deepcopy(img), conf_joints_ids)

        output_dir_2D = output_dir +'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)

        ## 3D
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose(post_out, ax, conf_joints_ids)

        output_dir_3D = output_dir +'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        plt.clf()
        plt.close(fig)
        
    print('Generating 3D pose successful!')

    output_npz = output_dir + '3d_keypoints_poseformer.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints_3d)

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.clf()
        plt.close(fig)