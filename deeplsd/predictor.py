from pathlib import Path

import cv2
import numpy as np
import torch

from deeplsd.models.deeplsd_inference import DeepLSD


class Predictor:
    def __init__(self, conf=None, ckpt=None):
        conf = conf or {
            'detect_lines': True,  # Whether to detect lines or only DF/AF
            'line_detection_params': {
                'merge': False,  # Whether to merge close-by lines
                # 'optimize': False,  # Whether to refine the lines after detecting them
                # 'use_vps': True,  # Whether to use vanishing points (VPs) in the refinement
                # 'optimize_vps': True,  # Whether to also refine the VPs in the refinement
                'filtering': True,
                # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
                'grad_thresh': 3,
                'grad_nfa': True,
                # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
            }
        }
        ckpt = ckpt or Path(__file__).parent / '../weights/deeplsd_md.tar'
        if not ckpt.exists():
            raise ValueError(f'Cannot find checkpoint at {ckpt}. Run `mkdir weights && wget https://www.polybox.ethz.ch/index.php/s/XVb30sUyuJttFys/download -O weights/deeplsd_md.tar` in the root of the repository to download the weights')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DeepLSD(conf)
        weights = torch.load(str(ckpt), map_location=self.device)
        self.net.load_state_dict(weights['model'], strict=False)
        self.net = self.net.to(self.device).eval()

    def set_threshold(self, new_thresh: float = 3.0):
        self.net.conf['line_detection_params']['grad_thresh'] = new_thresh

    def set_filtering(self, new_filtering: bool = True):
        self.net.conf['line_detection_params']['filtering'] = new_filtering

    def predict(self, image: np.ndarray):
        if len(image.shape) != 2:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = image
        inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=self.device)[None, None] / 255.}
        with torch.no_grad():
            out = self.net(inputs)
            pred_lines, = out['lines']
        return pred_lines

    @staticmethod
    def draw_lines(img: np.ndarray, lines: np.ndarray, color=(0, 255, 0)):
        for line in lines:
            (x1, y1), (x2, y2) = line
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
        return img
