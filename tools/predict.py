import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse
import cv2
import glob
import numpy as np
import paddle
from paddle.nn import functional as F

from models.rrdbnet_arch import RRDBNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../inputs/00003.png', help='Input image or folder')
    parser.add_argument(
        '--model_path',
        type=str,
        # default = '/home/aistudio/work/Real-ESRGAN-paddle1129/experiments/pretrained_models/RealESRGAN_x4plus.pdparams',
        default = '../experiments/pretrained_models_1/net_g_latest7.pdparams',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--block', type=int, default=23, help='num_block in RRDB')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args =  parser.parse_known_args()[0]

    if 'RealESRGAN_x4plus_anime_6B.pdparams' in args.model_path:
        args.block = 6
    elif 'RealESRGAN_x2plus.pdparams' in args.model_path:
        args.netscale = 2

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=args.block, num_grow_ch=32, scale=args.netscale)
    loadnet = paddle.load(args.model_path)
    model.set_state_dict(loadnet["params"])
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        scale_percent = 40       # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None
        
        img = img.astype(np.float32)
        img = paddle.to_tensor(np.transpose(img, (2, 0, 1)))
        out_img = img.unsqueeze(0)
        # pre_pad
        # out_img = F.pad(out_img, (0, 10, 0, 10), 'reflect')

        output = model(out_img)
        output_img = output.squeeze().numpy().clip(0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if args.ext == 'auto':
            extension = extension[1:]
        else:
            extension = args.ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        cv2.imwrite(save_path, output_img)


if __name__ == '__main__':
    main()
