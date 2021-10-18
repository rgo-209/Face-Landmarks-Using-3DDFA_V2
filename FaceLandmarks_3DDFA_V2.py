# -*- coding: utf-8 -*-
"""Facial Landmark extraction using 3DDFA_v2

This module demonstrates use of 3DDFA_V2 library 
for Facial Landmarks extraction.

Example
-------

    $ python3 FaceLandmarks_3DDFA_V2.py

Notes
-----
    Change the values of source video, feature file and output video file in main function.

"""

import logging
import sys
import imageio
import numpy as np
from tqdm import tqdm
import yaml
import cv2
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
import os
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
import warnings

warnings.filterwarnings('ignore')
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

sys.path.append('/home/ubuntu/development/general-tools')
from H5IO import h5_read, h5_write


class FaceLandmarks_3DDFA_V2:
    def __init__(self, opt='2d_sparse', use_GPU=False, config_file='configs/mb1_120x120.yml'):
        """Function to initialize a FaceLandmarks_3DDFA_V2
           class object.
        
        Parameters
        ----------
        opt : str
            The type of landmarks we want to fetch.
        use_GPU : bool
            Flag to use GPU or not.
        config_file : str
            The path to the config file to be used.

        Returns:
            None

        """
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        self.cfg = yaml.load(open(config_file), Loader=yaml.SafeLoader)

        self.use_GPU = use_GPU

        # type of landmark view ['2d_sparse','2d_dense']
        self.opt = opt

        if self.opt == '2d_sparse':
            self.n_landmarks = 68
        elif self.opt == '2d_dense':
            self.n_landmarks = 38365
        else:
            print('Wrong display option..!!')
            sys.exit(-1)

    def compute_features(self, video_path_in):
        """Function to compute the features using 3DDFA_V2 library.

        Parameters
        ----------
        video_path_in : str
            The path to the video file to be used.

        Returns:
            None

        """
        self.feats = None

        if self.use_GPU:
            # GPU and ONNX enabled
            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**self.cfg)
        else:
            # GPU and ONNX disabled
            face_boxes = FaceBoxes()
            tddfa = TDDFA(gpu_mode=False, **self.cfg)

        print(f'Running 3DDFA_V2 on {video_path_in}')

        # Given a video path generate reader
        reader = imageio.get_reader(video_path_in)

        cap = cv2.VideoCapture(video_path_in)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f'Number of frames {n_frames}')
        print(f'Number of landmarks {self.n_landmarks}')

        # [t0, t1, ldmk0_x, ldmk0_y, ldmk0_z, ldmk1_x, ldmk1_y, ldmk1_z, ...]
        feats = np.empty((n_frames, 2 + 3 * self.n_landmarks),
                         dtype=np.float32)
        feats[:] = np.nan

        self.fps = reader.get_meta_data()['fps']

        # run 3ddfa_v2
        dense_flag = self.opt is '2d_dense'

        pre_ver = None

        for i, frame in tqdm(enumerate(reader)):

            # convert frame RGB->BGR
            frame_bgr = frame[..., ::-1]
            
            # mirror
#             frame_bgr = frame[:, ::-1, :]

            if i == 0:
                # the first frame, detect face,
                # here we only use the first face,
                # you can change depending on your need
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                # refine
                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            else:
                param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

                roi_box = roi_box_lst[0]
                # todo: add confidence threshold to judge the tracking is failed
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame_bgr)
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # append to the landmarks
            feats[i, 0] = i / self.fps  # t0
            feats[i, 1] = i / self.fps  # t1

            for ld_idx, ldmk in enumerate(ver.T):
                feats[i, 2 + 3 * ld_idx + 0] = ldmk[0]
                feats[i, 2 + 3 * ld_idx + 1] = ldmk[1]
                feats[i, 2 + 3 * ld_idx + 2] = ldmk[2]
            # for tracking
            pre_ver = ver

        cap.release()

        self.feats = feats

    def save_features(self, feats_path):
        """Function to store the features to hdf5 file.
        
        Parameters
        ----------
        feats_path : str
            The path to the hdf5 file to be used.

        Returns:
            None

        """
        print(f'Writing features to {feats_path}')
        h5_write({'intervals': self.feats[:, :2], 'landmarks': self.feats[:, 2:]}, feats_path)
        print('Finished writing features.')

    def read_features(self, feats_path):
        """Function to read raw features from file.
        
        Parameters
        ----------
        feats_path : str
            The path to the hdf5 file to be used.

        Returns:
            None

        """
        print("Reading features from {}.".format(str(feats_path)))
        hdf5 = h5_read(feats_path)
        intrvl = hdf5['intervals']
        feats = hdf5['landmarks']
        print(intrvl.shape)
        print(feats.shape)
        print('Finished reading features.')

    def visualize_features(self, feats_path, video_path_in, video_path_out):
        """Function to generate the annotated video.
        
        Parameters
        ----------
        feats_path : str
            The path to the hdf5 file to be used.
        video_path_in : str
            The path to the source video file to be used.
        video_path_out : str
            The path to the output file to be saved.

        Returns:
            None

        """
        hdf5 = h5_read(feats_path)
        feats = hdf5['landmarks']

        cap = cv2.VideoCapture(video_path_in)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fr_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fr_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        out = cv2.VideoWriter(video_path_out, fourcc, fps, (int(fr_w), int(fr_h)))

        print('Generating visualization of features...')
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            # mirror
#             image = cv2.flip(image,1)
            
            pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            for i in range(self.n_landmarks):
                x = int(feats[pos_frames, 3 * i + 0])
                y = int(feats[pos_frames, 3 * i + 1])
                image[y - 1:y + 2, x - 1:x + 2, :] = 0
                image[y, x, :] = 255

            out.write(image)

        # cleanup
        cap.release()
        out.release()

        print(f'Saved video output to {video_path_out}')


if __name__ == '__main__':
    """This is the main function.
    
    """
    # video source
    video_path_in = '1DmNV9C1hbY.mp4'
    video_path_out = 'normal_1DmNV9C1hbY_3DDFA.mp4'
    feats_path = 'normal_1DmNV9C1hbY_3ddfa_features.h5'
    # config file source
    config_file = 'configs/mb1_120x120.yml'

    print("Config file of model used: ", config_file)
    print("Source Video file: ", video_path_in)
    print("Output Video file: ", video_path_out)
    print("Features file: ", feats_path)

    # FaceLandmarks_3DDFA_V2 object
    # type of landmark view ['2d_sparse','2d_dense']
    run_3ddfa_obj = FaceLandmarks_3DDFA_V2(opt='2d_sparse', use_GPU=False, config_file=config_file)

    # options: ['compute features', 'visualize features', read features', 'perform all']
    phase = 'perform all'

    if phase == 'compute features':
        # run the model
        run_3ddfa_obj.compute_features(video_path_in)
        run_3ddfa_obj.save_features(feats_path)

    if phase == 'read features':
        # source to read landmark hdf5 file
        run_3ddfa_obj.read_features(feats_path)

    if phase == 'visualize features':
        # call function to visualize
        run_3ddfa_obj.visualize_features(feats_path, video_path_in, video_path_out)

    if phase == 'perform all':
        run_3ddfa_obj.compute_features(video_path_in)
        run_3ddfa_obj.save_features(feats_path)

        # source to read landmark hdf5 file
        run_3ddfa_obj.read_features(feats_path)

        # call function to visualize
        run_3ddfa_obj.visualize_features(feats_path, video_path_in, video_path_out)
