"""Test the pretrained Pytorch model."""

import torch
import numpy as np

from model import InceptionI3d

USE_CUDA = False

def main():
    # Loading the model
    model = InceptionI3d(num_classes=400)
    model.eval()
    checkpoint = torch.load('weights/pt_rgb_imagenet/model.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # Loading the video.
    np_input = np.load('data/v_CricketShot_g04_c01_rgb.npy')
    import pdb; pdb.set_trace()

    # Reshaping the video for Pytorch (batch, channel, time, height, width)
    np_input = np.moveaxis(np_input, [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])

    input = torch.autograd.Variable(torch.FloatTensor(np_input))

    if USE_CUDA:
        model.cuda()
        input = input.cuda()

    out, logits = model(input)
    out = out.data.cpu().numpy().flatten()  # shape (400)
    logits = logits.data.cpu().numpy().flatten()  # shape (400)

    idx = np.argsort(out)
    top_preds_idx = idx[-10:][::-1]
    top_preds_output = out[top_preds_idx]

    print('Top predictions:')
    for idx, output in zip(top_preds_idx, top_preds_output):
        print(f'Class: {idx}, output: {output}, logits: {logits[idx]}')


if __name__ == '__main__':
    main()
