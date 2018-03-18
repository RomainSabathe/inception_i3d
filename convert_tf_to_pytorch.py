"""Loads the weights from the Inception i3d (Tensorflow version) model and
apply them to the Pytorch version of the model."""

import os
import re

import torch
import numpy as np
import tensorflow as tf

from model import InceptionI3d


def load_tf_weights():
    weights_path = 'weights/tf_rgb_imagenet/model.ckpt'
    reader = tf.train.NewCheckpointReader(weights_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    data = dict()
    for name in var_to_shape_map:
        print(name)
        tensor = reader.get_tensor(name)
        data[name] = tensor

    return data


def perform_regex_query(query, candidate):
    matches = re.findall(query, candidate)
    if len(matches) > 1:
        raise Exception('Ambiguity for candidate {} and query {} '
                        ' block.'.format(candidate, query))
    elif len(matches) == 1:
        return matches[0]
    return ''


def match_tf_name_with_pt_name(tf_name, state_dict):
    mixed_name = perform_regex_query(r'Mixed_[3-5][b-f]', tf_name)
    branch_name = perform_regex_query(r'Branch_[0-3]', tf_name)

    conv_name = perform_regex_query(r'Conv3d_\d[a-d]_\dx\d', tf_name)
    is_conv_op = '/conv_3d/' in tf_name
    is_weights = tf_name[-2:] == '/w'
    is_bias = tf_name[-2:] == '/b'

    # The Tensorflow model had an incorrect name for a specifc conv layer,
    # correcting this.
    if mixed_name == 'Mixed_5b' and branch_name == 'Branch_2' and \
       conv_name == 'Conv3d_0a_3x3':
           conv_name = 'Conv3d_0b_3x3'

    batch_norm = perform_regex_query(r'batch_norm', tf_name)
    is_batch_norm = batch_norm != ''
    is_moving_mean = 'batch_norm/moving_mean' in tf_name
    is_moving_variance = 'batch_norm/moving_variance' in tf_name
    is_bn_bias = 'batch_norm/beta' in tf_name

    candidates = []
    for key in state_dict.keys():
        if mixed_name not in key:
            continue
        if branch_name not in key:
            continue
        if conv_name not in key:
            continue
        if batch_norm not in key:
            continue
        if is_batch_norm:
            if is_moving_mean and 'running_mean' not in key:
                continue
            if is_moving_variance and 'running_var' not in key:
                continue
            if is_bn_bias and 'batch_norm.bias' not in key:
                continue
        elif is_conv_op:
            if is_weights and key[-13:] != 'conv3d.weight':
                continue
            if is_bias and key[-11:] != 'conv3d.bias':
                continue

        candidates.append(key)

    if len(candidates) > 1:
        raise Exception('Found multiple candidates when looking for {}'.format(
                        tf_name))
    if len(candidates) == 0:
        raise Exception('Found no candidate when looking for {}'.format(
                        tf_name))

    return candidates[0]


def reset_weights_batch_norm(state_dict):
    """Sonnet doesn't scale the batch-norm term by default. This has been the
    case for the original implementation of Inception3d. However,  when
    initializing the Pytorch version, we're using random scaling so we need
    to fix this."""

    for key in state_dict.keys():
        if 'batch_norm.weight' in key:
            param = state_dict[key]
            param_data = np.ones(param.shape)
            state_dict[key] = torch.nn.Parameter(torch.FloatTensor(param_data),
                                                 requires_grad=False)

    return state_dict


def save_pretrained_pt_model():
    dict_weights = load_tf_weights()
    model = InceptionI3d(num_classes=400)

    tf_names = dict_weights.keys()
    state_dict = model.state_dict()
    for tf_name in tf_names:
        pt_name = match_tf_name_with_pt_name(tf_name, state_dict)

        # Tensorflow and Pytorch don't use the same convention for shaping
        # the tensors. For 3d convolutional kernels, the convention is:
        # TF: [D, H, W, in_channels, out_channels]
        # PT: [out_channels, in_channels, D, H, W]
        param_data = dict_weights[tf_name]
        if len(param_data.shape) == 5:
            param_data = np.moveaxis(param_data,
                                     source=[0, 1, 2, 3, 4],
                                     destination=[2, 3, 4, 1, 0])

        param = state_dict[pt_name]
        assert param_data.shape == param.shape
        state_dict[pt_name] = torch.nn.Parameter(torch.FloatTensor(param_data))

    state_dict = reset_weights_batch_norm(state_dict)

    to_save = {'state_dict': state_dict,
               'version': 'RGB/ImageNet'}
    save_loc = 'weights/pt_rgb_imagenet/model.pth.tar'
    base_dir = os.path.dirname(save_loc)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torch.save(to_save, save_loc)


if __name__ == '__main__':
    save_pretrained_pt_model()
