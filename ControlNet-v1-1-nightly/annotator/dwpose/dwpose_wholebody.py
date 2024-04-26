import cv2
import os
import sys
import torch

import numpy as np
from . import util
from .wholebody import Wholebody
from . import draw_pose


class MyWholebody (Wholebody):
    def __init__(self, log_prefix, action, defaults_mod, defaults):

        device = defaults_mod.get("device", action, "cpu", defaults)
        print(f"{log_prefix} Specifying the device \"{device}\"")
        if device == "cpu":
            dev_name = "CPU"
        else:
            dev_name = torch.cuda.get_device_name()
        print(f"{log_prefix} Device: {dev_name}")

        backend = cv2.dnn.DNN_BACKEND_OPENCV if device == "cpu" else cv2.dnn.DNN_BACKEND_CUDA
        # You need to manually build OpenCV through cmake to work with your GPU.
        providers = cv2.dnn.DNN_TARGET_CPU if device == "cpu" else cv2.dnn.DNN_TARGET_CUDA

        base_dir = os.path.join(
            "modules",
            "extra",
            "dwpose"
        )
        print(f"{log_prefix} Base Directory: {base_dir}")

        onnx_det = f"{base_dir}/ckpts/yolox_l.onnx"
        onnx_pose = f"{base_dir}/ckpts/dw-ll_ucoco_384.onnx"

        if not os.path.isfile(onnx_det):
            print(f"{log_prefix} File {onnx_det} not found")
            sys.exit(1)

        if not os.path.isfile(onnx_pose):
            print(f"{log_prefix} File {onnx_det} not found")
            sys.exit(1)

        self.session_det = cv2.dnn.readNetFromONNX(onnx_det)
        self.session_det.setPreferableBackend(backend)
        self.session_det.setPreferableTarget(providers)

        self.session_pose = cv2.dnn.readNetFromONNX(onnx_pose)
        self.session_pose.setPreferableBackend(backend)
        self.session_pose.setPreferableTarget(providers)


class DWposeDetector:
    def __init__(self, log_prefix, action, defaults_mod, defaults):

        self.pose_estimation = MyWholebody(
            log_prefix=log_prefix,
            action=action,
            defaults_mod=defaults_mod,
            defaults=defaults,
        )

        self.log_prefix = log_prefix
        self.action = action
        self.defaults_mod = defaults_mod
        self.defaults = defaults

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces, foot=foot)

            self.pose = pose

            return self.draw_pose(pose, H, W)

    def draw_hand(self, pose, H, W):

        hands = pose['hands']
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        print(f"{self.log_prefix} Drawing hand")
        canvas = util.draw_handpose(canvas, hands)

        return canvas

    def draw_pose(self, pose, H, W, **kwargs):

        bodies = pose['bodies']
        faces = pose['faces']
        hands = pose['hands']
        foot = pose['foot']

        candidate = bodies['candidate']
        subset = bodies['subset']
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        draw_body = self.defaults_mod.get(
            "dwpose_draw_body",
            self.action,
            True,
            self.defaults,
        )
        if kwargs.get("no_body", False):
            draw_body = False

        draw_hand = self.defaults_mod.get(
            "dwpose_draw_hand",
            self.action,
            True,
            self.defaults,
        )
        if kwargs.get("no_hand", False):
            draw_hand = False

        draw_face = self.defaults_mod.get(
            "dwpose_draw_face",
            self.action,
            True,
            self.defaults,
        )
        if kwargs.get("no_face", False):
            draw_face = False

        draw_foot = self.defaults_mod.get(
            "dwpose_draw_foot",
            self.action,
            True,
            self.defaults,
        )
        if kwargs.get("no_foot", False):
            draw_foot = False

        if draw_body:
            print(f"{self.log_prefix} Drawing body")
            canvas = util.draw_bodypose(canvas, candidate, subset)

        if draw_hand:
            print(f"{self.log_prefix} Drawing hand")
            canvas = util.draw_handpose(canvas, hands)

        if draw_face:
            print(f"{self.log_prefix} Drawing face")
            canvas = util.draw_facepose(canvas, faces)

        if draw_foot:
            print(f"{self.log_prefix} Drawing foot")
            canvas = util.draw_facepose(canvas, foot)

        return canvas
