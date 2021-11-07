# evaluation_curvatures.py run curvature estimation evaluation
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

import os
import numpy as np
import pickle
import utils
import argparse

# python evaluate_curvatures.py --dataset_list testset_temp --sparse_patches=1
def l2_norm(v):
    norm_v = np.sqrt(np.sum(np.square(v), axis=1))
    return norm_v

def map_curvatures1(current_curvatures):
    """
    map_curvatures maps the curvature values of the input to be minimum and maximum and disregards sign (upper triangle)
    :param current_curvatures: B x 2 principal curvature values to map
    :return: B x 2 mapped curvature values
    """
    # maps the curvatures to unoriented maximum and minimum curvatures
    mapped_curvatures = np.zeros_like(current_curvatures)
    min_c = current_curvatures.min(axis=1)
    max_c = current_curvatures.max(axis=1)
    mapped_curvatures[:, 0] = min_c
    mapped_curvatures[:, 1] = max_c
    idxs = mapped_curvatures[:, 1] < -mapped_curvatures[:, 0]
    mapped_curvatures[idxs, 0] = -max_c[idxs]
    mapped_curvatures[idxs, 1] = -min_c[idxs]
    return mapped_curvatures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/sitzikbs/Datasets/pcpnet/', help='Relative path to data directory')
parser.add_argument('--sparse_patches', default=True, help='sparse patches indicator, choose True for pcpnet evaluation'
                                                           ',False does not apply if test was not full')
parser.add_argument('--results_path', default='./log/baselines/DeepFit/results/', help='path to trained model')
parser.add_argument('--map_curvatures', type=int, default=True, help='map curvatures indicator')
parser.add_argument('--dataset_list', type=str, nargs='+',
                    default=['testset_no_noise', 'testset_low_noise', 'testset_med_noise', 'testset_high_noise',
                              'testset_vardensity_striped', 'testset_vardensity_gradient'],
                    help='choose file lists to run evaluation on')

FLAGS = parser.parse_args()

MAP_CURVATURES = FLAGS.map_curvatures
sparse_patches = FLAGS.sparse_patches
dataset_list = FLAGS.dataset_list
PC_PATH = os.path.join(BASE_DIR, FLAGS.data_path)
results_path = os.path.join(BASE_DIR, FLAGS.results_path)

for dataset in dataset_list:
    curv_results_path = results_path  # for older runs
    curv_gt_filenames = PC_PATH + dataset + '.txt'

    curvatures_gt_path = PC_PATH
    # get all shape names in the dataset
    shape_names = []
    with open(curv_gt_filenames) as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    summary_dir_name = 'summary/'
    if MAP_CURVATURES:
        summary_dir_name = 'summary_mapped/'
    outdir = os.path.join(curv_results_path, summary_dir_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    LOG_FOUT = open(os.path.join(outdir, dataset + '_curv_evaluation_results.txt'), 'w')


    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)


    experts_exist = False
    rms_regular = []
    true_rms_L = []
    rms_L = []
    rms_tanh = []
    for i, shape in enumerate(shape_names):
        print('Processing ' + shape + '...')

        # load the data
        points = np.loadtxt(os.path.join(curvatures_gt_path, shape + '.xyz')).astype('float32')
        curvatures_gt = np.loadtxt(os.path.join(curvatures_gt_path, shape + '.curv')).astype('float32')
        curvatures_results = np.loadtxt(os.path.join(curv_results_path, shape + '.curv')).astype('float32')
        points_idx = np.loadtxt(os.path.join(curvatures_gt_path, shape + '.pidx')).astype('int')

        normals_gt = np.loadtxt(os.path.join(curvatures_gt_path, shape + '.normals')).astype('float32')
        normals_results = np.loadtxt(os.path.join(curv_results_path, shape + '.normals')).astype('float32')

        if os.path.exists(os.path.join(curv_results_path, shape + '.experts')):
            experts_exist = True
            experts = np.loadtxt(os.path.join(curv_results_path, shape + '.experts'))
            params = pickle.load(open(results_path + 'parameters.p', "rb"))
            n_experts = params.n_experts

        n_points = points.shape[0]
        n_curvatures = curvatures_results.shape[0]

        if sparse_patches:
            points = points[points_idx, :]
            curvatures_gt = curvatures_gt[points_idx, :]
            normals_gt = normals_gt[points_idx, :]
            normals_results = normals_results[points_idx, :]

        if n_points != n_curvatures:
            sparse_curvatures = True
        else:
            sparse_curvatures = False

        if sparse_patches and not sparse_curvatures:
            curvatures_results = curvatures_results[points_idx, :]


        if (not sparse_patches) and sparse_curvatures:
            raise ValueError('Inconsistent sparse patches request - rerun test with sparse_patches set to False')

        # jet stored curvatures with an additional value - this is to remove it
        if curvatures_results.shape[1] > 2:
            curvatures_results = curvatures_results[:, 0:2]

        # flip the sign according to the normal
        sign = np.sign(np.sum(normals_results * normals_gt, axis=1))
        curvatures_results = curvatures_results * np.tile(sign, [2, 1]).transpose()

        if MAP_CURVATURES: #first column maximum, second minimum
            curvatures_results = map_curvatures(curvatures_results)  # for pcpnet


        # Not oriented rms
        diff_c = curvatures_results - curvatures_gt
        rms_regular_shape = np.sqrt(np.nanmean(np.square(diff_c), axis=0))
        true_rms_L_shape = np.sqrt(np.nanmean(np.square(diff_c/np.maximum(np.abs(curvatures_gt), np.ones_like(curvatures_gt))), axis=0))
        rms_L_shape = np.nanmean(np.abs((diff_c / np.maximum(np.abs(curvatures_gt), np.ones_like(curvatures_gt)))),
                                 axis=0)
        expanssion_coeff = 0.1
        rms_tanh_shape = np.sqrt(np.nanmean(
            np.square(np.tanh(expanssion_coeff * curvatures_results) - np.tanh(expanssion_coeff * curvatures_gt)),
            axis=0))

        # error metrics
        rms_regular.append(rms_regular_shape)
        true_rms_L.append(true_rms_L_shape)
        rms_L.append(rms_L_shape)
        rms_tanh.append(rms_tanh_shape)


    avg_rms_regular = np.mean(rms_regular, axis=0)
    avg_true_rms_L = np.mean(true_rms_L, axis=0)
    avg_rms_L = np.mean(rms_L, axis=0)
    avg_rms_tanh = np.mean(rms_tanh, axis=0)
    # avg_rms_o = np.mean(rms_o)
    rms_regular = np.array(rms_regular)
    true_rms_L = np.array(true_rms_L)
    rms_L = np.array(rms_L)
    rms_tanh = np.array(rms_tanh)

    log_string('k1 regular RMS per shape: ' + str(rms_regular[:, 0]))
    log_string('k2 regular RMS per shape: ' + str(rms_regular[:, 1]))
    log_string('k1 regular average RMS: ' + str(avg_rms_regular[0]))
    log_string('k2 regular average RMS: ' + str(avg_rms_regular[1]) + '\n')

    log_string('k1 L RMS per shape: ' + str(rms_L[:, 0]))
    log_string('k2 L RMS per shape: ' + str(rms_L[:, 1]))
    log_string('k1 L average RMS: ' + str(avg_rms_L[0]))
    log_string('k2 L average RMS: ' + str(avg_rms_L[1]) + '\n')

    log_string('k1 L true RMS per shape: ' + str(true_rms_L[:, 0]))
    log_string('k2 L true RMS per shape: ' + str(true_rms_L[:, 1]))
    log_string('k1 L true average RMS: ' + str(avg_true_rms_L[0]))
    log_string('k2 L true average RMS: ' + str(avg_true_rms_L[1]) + '\n')

    log_string('k1 tanh RMS per shape: ' + str(rms_tanh[:, 0]))
    log_string('k2 tanh RMS per shape: ' + str(rms_tanh[:, 1]))
    log_string('k1 tanh average RMS: ' + str(avg_rms_tanh[0]))
    log_string('k2 tanh average RMS: ' + str(avg_rms_tanh[1]) + '\n')

    LOG_FOUT.close()
