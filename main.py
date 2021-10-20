# -*- coding: utf-8 -*-
"""Facial Landmark extraction using 3DDFA_v2
"""

import logging
import sys
import imageio
import numpy as np
import pandas as pd
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

class TDDFA_Impl:
    def __init__(self, in_video_path, out_csv_path='test.csv', out_video_path='', opt='2d_sparse', use_GPU=False,
                 config_file='configs/mb1_120x120.yml'):
        """Function to initialize a TDDFA_Impl class object.
        
        Parameters
        ----------
        in_video_path : str
            the path to the video to be used
        out_csv_path : str
            the path to the output csv file path
        out_video_path : str
            the path to the video to be used
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

        self.in_video_path = in_video_path
        self.out_video_path = out_video_path
        self.out_csv_path = out_csv_path

        self.columns = ['frame_num']
        # store coordinates as
        # [frame_num, x_0, y_0, z_0,  x_1, y_1, z_1, .....  x_467, y_467, z_467 ]
        for i in range(self.n_landmarks):
            self.columns.append('x_' + str(i))
            self.columns.append('y_' + str(i))
            self.columns.append('z_' + str(i))

    def generate_features(self):
        """Function to compute the landmarks using 3DDFA_V2
            and stores visualizations if required.
        Returns:
            None
        """

        # set the flag if 3DDFA should use GPU or not
        if self.use_GPU:
            # GPU and ONNX enabled
            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**self.cfg)
        else:
            # GPU and ONNX disabled
            face_boxes = FaceBoxes()
            tddfa = TDDFA(gpu_mode=False, **self.cfg)

        # Given a video path generate reader
        reader = imageio.get_reader(self.in_video_path)

        cap = cv2.VideoCapture(self.in_video_path)

        # execute following when you want to create visuals of the video
        if self.out_video_path != '':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            fr_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fr_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # create writer object for output video
            out = cv2.VideoWriter(self.out_video_path, fourcc, fps, (int(fr_w), int(fr_h)))

        print(f'Running 3DDFA_V2 on {self.in_video_path}')

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f'Number of frames {n_frames}')
        print(f'Number of landmarks {self.n_landmarks}')

        # [t0, ldmk0_x, ldmk0_y, ldmk0_z, ldmk1_x, ldmk1_y, ldmk1_z, ...]
        feats = np.empty((n_frames, 1 + 3 * self.n_landmarks),
                         dtype=np.float32)
        feats[:] = np.nan

        dense_flag = self.opt is '2d_dense'

        pre_ver = None

        for i, frame in tqdm(enumerate(reader)):

            # convert frame RGB->BGR
            frame_bgr = frame[..., ::-1]

            # mirror
            # frame = frame[:, ::-1, :]

            if i == n_frames:
                break

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
            feats[i, 0] = i

            for ld_idx, ldmk in enumerate(ver.T):
                feats[i, 1 + 3 * ld_idx + 0] = ldmk[0]
                feats[i, 1 + 3 * ld_idx + 1] = ldmk[0]
                feats[i, 1 + 3 * ld_idx + 2] = ldmk[2]

                # plot points on the original image
                # and write to output video
                if self.out_video_path != '':
                    x = int(ldmk[0])
                    y = int(ldmk[1])
                    frame[y - 1:y + 2, x - 1:x + 2, :] = 0
                    frame[y, x, :] = 255

            # write the frame to output video
            out.write(frame)

            # for tracking
            pre_ver = ver

        cap.release()

        if self.out_video_path != '':
            print("Stored visualizations to %s successfully !!" % self.out_video_path)
            out.release()

        print("Generated landmarks for the video %s" % self.in_video_path)
        print("Writing facial landmarks for the video to %s " % self.out_csv_path)

        data = pd.DataFrame(feats, columns=self.columns)
        data.to_csv(self.out_csv_path, index=False)
        print("Saved the landmarks for the video to %s  !!" % self.out_csv_path)


if __name__ == '__main__':
    """This is the main function.
    
    """
    # video source
    in_vid_path = 'Testing/test_vid.mp4'

    # output video with points plotted on face
    # pass empty string if no visualization to be generated
    out_vid_path = 'Testing/test_vid_3ddfa_op.mp4'

    # path to csv file for storing landmarks data
    out_csv_path = 'Testing/test_vid_3ddfa_features.csv'

    # config file source
    config_file = 'configs/mb1_120x120.yml'

    print("Config file of model used: ", config_file)
    print("Source Video file: ", in_vid_path)
    print("Output Video file: ", out_vid_path)
    print("Features file: ", out_csv_path)

    # type of landmark view are ['2d_sparse','2d_dense']
    opt = '2d_sparse'

    # flag for GPU usage
    use_GPU = False

    # create object of TDDFA_Impl class with params
    tddfa_obj = TDDFA_Impl(in_vid_path, out_csv_path, out_vid_path, opt, use_GPU, config_file)

    # generate features and create visualizations
    tddfa_obj.generate_features()
