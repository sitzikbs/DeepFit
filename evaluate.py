# evaluation.py run normal estimation evaluation
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

import os
import numpy as np
import pickle
import argparse
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import visualization
import utils


def l2_norm(v):
    norm_v = np.sqrt(np.sum(np.square(v), axis=1))
    return norm_v

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASELINE_DIR, os.pardir))

# Exection:
# python utils/evaluate.py --normal_results_path='log/experts/pcpnet_results/' --data_path='data/pcpnet/' --dataset_list 'testset' 'testset_whitenoise_small' 'testset_whitenoise_medium' 'testset_whitenoise_large' 'testset_vardensity_gradient' 'testset_vardensity_striped'
# python3 evaluate.py --normal_results_path './log/jetnet_nci_new3/ablations/Deepfit_simple_sigmoid_cr_log_d1_p64_Lsin/results/'
parser = argparse.ArgumentParser()
parser.add_argument('--normal_results_path', type=str, default='./log/baselines/Lenssen et. al./', help='Log dir [default: log]')
parser.add_argument('--data_path', type=str, default='/home/sitzikbs/Datasets/pcpnet/', help='Relative path to data directory')
parser.add_argument('--sparse_patches', type=int, default=True, help='Evaluate on a sparse subset or on the entire point cloud')
parser.add_argument('--dataset_list', type=str,
                    default=['testset_no_noise', 'testset_low_noise', 'testset_med_noise', 'testset_high_noise',
                              'testset_vardensity_striped', 'testset_vardensity_gradient'], nargs='+',
                    help='list of .txt files containing sets of point cloud names for evaluation')
FLAGS = parser.parse_args()
EXPORT = False  # export some visualizations
PC_PATH = os.path.join(BASE_DIR, FLAGS.data_path)
normal_results_path = FLAGS.normal_results_path
results_path = os.path.abspath(os.path.join(normal_results_path, os.pardir))
sparse_patches = FLAGS.sparse_patches

if not os.path.exists(normal_results_path):
    ValueError('Incorrect normal results path...')

dataset_list = FLAGS.dataset_list

for dataset in dataset_list:

    normal_gt_filenames = PC_PATH + dataset + '.txt'

    normal_gt_path = PC_PATH

    # get all shape names in the dataset
    shape_names = []
    with open(normal_gt_filenames) as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    outdir = os.path.join(normal_results_path, 'summary/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    LOG_FOUT = open(os.path.join(outdir, dataset + '_evaluation_results.txt'), 'w')
    if EXPORT:
        file_path = os.path.join(normal_results_path, 'images')
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    experts_exist = False
    rms = []
    rms_o = []
    all_ang = []
    pgp10 = []
    pgp5 = []
    pgp_alpha = []

    for i, shape in enumerate(shape_names):
        print('Processing ' + shape + '...')

        if EXPORT:
            # Organize the output folders
            idx_1 = shape.find('_noise_white_')
            idx_2 = shape.find('_ddist_')
            if idx_1 == -1 and idx_2 == -1:
                base_shape_name = shape
            elif idx_1 == -1:
                base_shape_name = shape[:idx_2]
            else:
                base_shape_name = shape[:idx_1]

            vis_output_path = os.path.join(file_path, base_shape_name)
            if not os.path.exists(vis_output_path):
                os.makedirs(vis_output_path)
            gt_normals_vis_output_path = os.path.join(vis_output_path, 'normal_gt')
            if not os.path.exists(gt_normals_vis_output_path):
                os.makedirs(gt_normals_vis_output_path)
            pred_normals_vis_output_path = os.path.join(vis_output_path, 'normal_pred')
            if not os.path.exists(pred_normals_vis_output_path):
                os.makedirs(pred_normals_vis_output_path)
            phi_teta_vis_output_path = os.path.join(vis_output_path, 'phi_teta_domain')
            if not os.path.exists(phi_teta_vis_output_path):
                os.makedirs(phi_teta_vis_output_path)

        # load the data
        points = np.loadtxt(os.path.join(normal_gt_path, shape + '.xyz')).astype('float32')
        normals_gt = np.loadtxt(os.path.join(normal_gt_path, shape + '.normals')).astype('float32')
        normals_results = np.loadtxt(os.path.join(normal_results_path, shape + '.normals')).astype('float32')
        points_idx = np.loadtxt(os.path.join(normal_gt_path, shape + '.pidx')).astype('int')

        n_points = points.shape[0]
        n_normals = normals_results.shape[0]
        if n_points != n_normals:
            sparse_normals = True
        else:
            sparse_normals = False

        points = points[points_idx, :]
        normals_gt = normals_gt[points_idx, :]
        # curvs_gt = curvs_gt[points_idx, :]
        if sparse_patches and not sparse_normals:
            normals_results = normals_results[points_idx, :]
        else:
            normals_results = normals_results[:, :]

        normal_gt_norm = l2_norm(normals_gt)
        normal_results_norm = l2_norm(normals_results)
        normals_results = np.divide(normals_results, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normals_gt = np.divide(normals_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        # Not oriented rms
        nn = np.sum(np.multiply(normals_gt, normals_results), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))  #  unoriented

        # error metrics
        rms.append(np.sqrt(np.mean(np.square(ang))))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))  # portion of good points
        pgp5_shape = sum([j < 5.0 for j in ang]) / float(len(ang))  # portion of good points
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)
        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))

        pgp_alpha.append(pgp_alpha_shape)

        # Oriented rms
        rms_o.append(np.sqrt(np.mean(np.square(np.rad2deg(np.arccos(nn))))))

        diff = np.arccos(nn)
        diff_inv = np.arccos(-nn)
        unoriented_normals = normals_results
        unoriented_normals[diff_inv < diff, :] = -normals_results[diff_inv < diff, :]

    avg_rms = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5 = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    LOG_FOUT.close()
