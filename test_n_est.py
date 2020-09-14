# test_n_est.py run a pretrained DeepFit model and export the normal estimation output
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

from __future__ import print_function
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import importlib.util
import time

from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_PATH = Path(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR_PATH, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from dataset import PointcloudPatchDataset, SequentialShapeRandomPointcloudPatchSampler, RandomPointcloudPatchSampler, SequentialPointcloudPatchSampler

# Execution
# python3 test_n_est.py --models 'Deepfit_simple_sigmoid_cr_log_d1_p64_Lsin' 'Deepfit_simple_sigmoid_cr_log_d2_p64_Lsin' 'Deepfit_simple_sigmoid_cr_log_d3_p64_Lsin' 'Deepfit_simple_sigmoid_cr_log_d4_p64_Lsin' 'Deepfit_simple_sigmoid_cr_log_d4_p128_Lsin' 'Deepfit_simple_sigmoid_cr_log_d3_p128_Lsin' 'Deepfit_simple_sigmoid_cr_log_d2_p128_Lsin' --logdir './log/jetnet_nci_new3/ablations/' --sparse_patches 1 --testset 'testset_all.txt'

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='/home/sitzikbs/Datasets/pcpnet/', help='input folder (point clouds)')
    parser.add_argument('--testset', type=str, default='testset_all.txt', help='shape set file name')
    parser.add_argument('--models', type=str, default='DeepFit_no_noise', help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_model_500.pth', help='model file postfix')
    parser.add_argument('--logdir', type=str, default='./log/', help='model folder')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--gpu_idx', type=int, default=1, help='set < 0 to use CPU')

    parser.add_argument('--sparse_patches', type=int, default=True, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    return parser.parse_args()

def test_n_est(opt):

    opt.models = opt.models.split()

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_idx)
    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % 0)

    for model_name in opt.models:
       # fetch the model from the log dir

        # append model name to output directory and create directory if necessary
        model_log_dir =  os.path.join(opt.logdir, model_name, 'trained_models')
        model_filename = os.path.join(model_log_dir, model_name+opt.modelpostfix)
        param_filename = os.path.join(model_log_dir, model_name+opt.parmpostfix)
        output_dir = os.path.join(opt.logdir, model_name, 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Random Seed: %d" % (opt.seed))
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        # load model and training parameters

        trainopt = torch.load(param_filename)
        if not hasattr(trainopt, 'arch'):
            trainopt.arch = 'simple'

    if opt.batchSize == 0:
        opt.batchSize = trainopt.batchSize

        # get indices in targets and predictions corresponding to each output
        target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim = get_target_features((trainopt))
        dataloader, dataset, datasampler = get_data_loaders(opt, trainopt, target_features)

        if trainopt.arch == 'simple':
            spec = importlib.util.spec_from_file_location("DeepFit", os.path.join(
                os.path.join(opt.logdir, model_name, "DeepFit.py")))
            DeepFit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(DeepFit)
            regressor = DeepFit.DeepFit(1, num_points=trainopt.points_per_patch,
                                                    use_point_stn=trainopt.use_point_stn,
                                                    use_feat_stn=trainopt.use_feat_stn, point_tuple=1,
                                                    sym_op=trainopt.sym_op, jet_order=trainopt.jet_order,
                                                    weight_mode=trainopt.weight_mode).cuda()
        elif trainopt.arch == '3dmfv':
            spec = importlib.util.spec_from_file_location("DeepFitNormals", os.path.join(
                os.path.join(opt.logdir, model_name, "DeepFitNormals.py")))
            DeepFit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(DeepFit)
            regressor = DeepFit.DeepFit(1, num_points=trainopt.points_per_patch,
                                                use_point_stn=trainopt.use_point_stn,
                                                use_feat_stn=trainopt.use_feat_stn, point_tuple=trainopt.point_tuple,
                                                sym_op=trainopt.sym_op, arch=trainopt.arch,
                                                n_gaussians=trainopt.n_gaussians, jet_order=trainopt.jet_order,
                                                weight_mode=trainopt.weight_mode).cuda()
        else:
            raise ValueError("unsupported architecture")

        regressor.load_state_dict(torch.load(model_filename))
        regressor.to(device)
        regressor.eval()

        shape_ind = 0
        shape_patch_offset = 0
        if opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

        num_batch = len(dataloader)
        batch_enum = enumerate(dataloader, 0)

        shape_ind = 0
        normal_prop = torch.zeros([shape_patch_count, 3])

        for batchind, data in batch_enum:

            # get  batch and upload to GPU
            points = data[0]
            target = data[1:-2]
            data_trans = data[-2]
            n_effective_points = data[-1].squeeze()

            points = points.transpose(2, 1)
            points = points.to(device)
            data_trans = data_trans.to(device)
            target = tuple(t.to(device) for t in target)
            start_time = 0
            end_time = 0
            with torch.no_grad():
                if trainopt.arch == 'simple' or trainopt.arch == 'res' or trainopt.arch == '3dmfv':
                    start_time = time.time()
                    n_est, beta_pred, weights, trans, trans2, neighbor_normals = regressor(points)
                    end_time = time.time()

            print("elapsed_time per point: {} ms".format(1000*(end_time-start_time) / opt.batchSize))

            if trainopt.use_point_stn:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                n_est[:, :] = torch.bmm(n_est.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)

            if trainopt.use_pca:
                # transform predictions with inverse pca rotation (back to world space)
                n_est[:, :] = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

            print('[%s %d/%d] shape %s' % (model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))

            # Save estimated normals to file
            batch_offset = 0

            while batch_offset < n_est.shape[0] and shape_ind + 1 <= len(dataset.shape_names):
                shape_patches_remaining = shape_patch_count - shape_patch_offset
                batch_patches_remaining = n_est.shape[0] - batch_offset

                # append estimated patch properties batch to properties for the current shape on the CPU
                normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining,
                                                                        batch_patches_remaining), :] = \
                    n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

                batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                if shape_patches_remaining <= batch_patches_remaining:
                    normals_to_write = normal_prop.cpu().numpy()
                    eps=1e-6
                    normals_to_write[np.logical_and(normals_to_write < eps, normals_to_write > -eps)] = 0.0
                    np.savetxt(os.path.join(output_dir, dataset.shape_names[shape_ind] + '.normals'),
                               normals_to_write)


                    print('saved normals for ' + dataset.shape_names[shape_ind])
                    sys.stdout.flush()
                    shape_patch_offset = 0
                    shape_ind += 1
                    if shape_ind < len(dataset.shape_names):
                        shape_patch_count = dataset.shape_patch_count[shape_ind]
                        normal_prop = torch.zeros([shape_patch_count, 3])


def get_data_loaders(opt, trainopt, target_features):
    # create dataset loader
    if opt.batchSize == 0:
        model_batchSize = trainopt.batchSize
    else:
        model_batchSize = opt.batchSize

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=trainopt.patch_radius,
        points_per_patch=trainopt.points_per_patch,
        patch_features=target_features,
        point_count_std=trainopt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=trainopt.identical_epochs,
        use_pca=trainopt.use_pca,
        center=trainopt.patch_center,
        point_tuple=trainopt.point_tuple,
        sparse_patches=opt.sparse_patches,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=trainopt.neighbor_search)
    if opt.sampling == 'full':
        test_datasampler = SequentialPointcloudPatchSampler(test_dataset)
    elif opt.sampling == 'sequential_shapes_random_patches':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=model_batchSize,
        num_workers=int(opt.workers))
    return test_dataloader, test_dataset, test_datasampler


def get_target_features(opt):
    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        elif o == 'neighbor_normals':
            target_features.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    return target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim


if __name__ == '__main__':
    eval_opt = parse_arguments()
    test_n_est(eval_opt)
