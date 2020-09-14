# train_n_est.py train a DeepFit model
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch

from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_PATH = Path(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR_PATH, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import normal_estimation_utils
import DeepFit

from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler, SequentialPointcloudPatchSampler

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='DeepFit_no_noise', help='training run name')
    parser.add_argument('--arch', type=str, default='simple', help='arcitecture name:  "simple" | "3dmfv"')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/home/sitzikbs/Datasets/pcpnet/', help='input folder (point clouds)')
    parser.add_argument('--logdir', type=str, default='./log/', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_no_noise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_no_noise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    parser.add_argument('--refine', action="store_true", help='flag to refine the model, path determined by outri and model name')
    parser.add_argument('--refine_epoch', type=int, default=500, help='refine model from this epoch')
    parser.add_argument('--overwrite', action="store_true", help='to overwrite existing log directory')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer adam / SGD / rmsprop')
    parser.add_argument('--opt_eps', type=float, default=1e-08, help='optimizer epsilon')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--scheduler_type', type=str, default='step', help='step or plateau')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--normal_loss', type=str, default='sin', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)\n'
                        'sin: mean sin(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals', 'neighbor_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')

    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--use_pca', type=int, default=True, help='use pca on point clouds, must be true for jet fit type')


    parser.add_argument('--n_gaussians', type=int, default=3, help='use feature spatial transformer')

    parser.add_argument('--jet_order', type=int, default=2, help='jet polynomial fit order')
    parser.add_argument('--points_per_patch', type=int, default=128, help='max. number of points per patch')
    parser.add_argument('--neighbor_search', type=str, default='k', help='[k | r] for k nearest and radius')
    parser.add_argument('--weight_mode', type=str, default="sigmoid", help='which function to use on the weight output: softmax, tanh, sigmoid')
    parser.add_argument('--use_consistency', type=int, default=True, help='flag to use consistency loss')
    parser.add_argument('--con_reg', type=str, default='log', help='choose consistency regularizer: mean, uniform')
    return parser.parse_args()


def log_string(out_str, log_file):
    log_file.write(out_str+'\n')
    log_file.flush()
    print(out_str)

def train_pcpnet(opt):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_idx)
    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % 0)
    # device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    out_dir = os.path.join(log_dirname, 'trained_models')
    params_filename = os.path.join(out_dir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(out_dir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(out_dir, '%s_description.txt' % (opt.name))
    log_filename = os.path.join(log_dirname, 'out.log')

    if (os.path.exists(log_dirname) or os.path.exists(model_filename)) and not opt.name == 'DeepFit_trainall' and opt.refine == '':
        if opt.overwrite:
            response = 'y'
        else:
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
        if response == 'y':
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))
        else:
            sys.exit()

    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))
    log_file = open(log_filename, 'w')

    model = get_model(opt, log_dirname)


    os.system('cp train_n_est.py %s' % (log_dirname))  # backup the current training file

    if opt.refine:
        refine_model_filename = os.path.join(out_dir, '{}_model_{}.pth' .format(opt.name, opt.refine_epoch))
        model.load_state_dict(torch.load(refine_model_filename))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    target_features, output_target_ind, output_pred_ind, output_loss_weight = get_target_features((opt))
    train_dataloader, train_dataset, train_datasampler, test_dataloader, test_dataset, \
    test_datasampler = get_data_loaders(opt, target_features)

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    log_string('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)), log_file)

    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0000001, eps=opt.opt_eps)
    elif opt.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=0.0000001, eps=opt.opt_eps)
    else:
        raise ValueError("Unsupported optimizer")

    if opt.scheduler_type == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500, 700], gamma=0.1) # milestones in number of optimizer iterations
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                               cooldown=5, min_lr=1e-012, eps=1e-08)
    model.to(device)

    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    refine_flag = True
    for epoch in range(opt.nepoch):
        if epoch <= opt.refine_epoch and opt.refine and refine_flag and opt.scheduler_type=='step':
            scheduler.step()
            continue
        else:
            refine_flag = False

        train_enum = enumerate(train_dataloader, 0)

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)
        avg_test_loss = 0.0
        for train_batchind, data in train_enum:

            # set to training mode
            model.train()

            # get trainingset batch and upload to GPU
            points = data[0]
            target = data[1:-2]
            # n_effective_points = data[-1].squeeze()

            points = points.transpose(2, 1)
            points = points.to(device)
            # data_trans = data_trans.to(device)
            target = tuple(t.to(device) for t in target)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            pred, beta_pred, weights, trans, trans2, neighbor_normals = model(points)

            loss, n_loss, _, consistency_loss, normal_loss = compute_loss(
                pred=pred, target=target,
                outputs=opt.outputs,
                output_pred_ind=output_pred_ind,
                output_target_ind=output_target_ind,
                output_loss_weight=output_loss_weight,
                normal_loss_type=opt.normal_loss,
                arch=opt.arch,
                patch_rot=trans if opt.use_point_stn else None,
                use_consistency=opt.use_consistency, point_weights=weights, neighbor_normals=neighbor_normals,
                opt=opt, trans=trans, trans2=trans2)

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            # print info and update log file
            log_string('[%s %d: %d/%d] %s loss: %f' % (opt.name, epoch,
                                                       train_batchind, train_num_batch-1, green('train'), loss.item()), log_file)
            train_writer.add_scalar('total_loss', loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
            train_writer.add_scalar('n_loss', n_loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
            train_writer.add_scalar('consistency_loss', consistency_loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
            train_writer.add_scalar('normal_loss', normal_loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
            test_writer.add_histogram('weights', weights.detach().cpu().numpy(),
                                      (epoch + train_fraction_done) * train_num_batch * opt.batchSize)

            while test_fraction_done <= train_fraction_done and test_batchind+1 < test_num_batch:

                # set to evaluation mode
                model.eval()

                test_batchind, data = next(test_enum)

                # get testset batch and upload to GPU
                points = data[0]
                target = data[1:-2]
                data_trans = data[-2]
                # n_effective_points = data[-1].squeeze()

                points = points.transpose(2, 1)
                points = points.to(device)
                data_trans = data_trans.to(device)

                target = tuple(t.to(device) for t in target)
                # weights = None

                # forward pass
                with torch.no_grad():
                    pred, beta_pred, weights, trans, trans2, neighbor_normals = model(points)

                loss, n_loss, err_angle, consistency_loss, normal_loss = compute_loss(
                    pred=pred, target=target,
                    outputs=opt.outputs,
                    output_pred_ind=output_pred_ind,
                    output_target_ind=output_target_ind,
                    output_loss_weight=output_loss_weight,
                    normal_loss_type=opt.normal_loss,
                    arch=opt.arch,
                    patch_rot=trans if opt.use_point_stn else None, phase='test',
                    use_consistency=opt.use_consistency, point_weights=weights, neighbor_normals=neighbor_normals,
                    opt=opt, trans=trans, trans2=trans2)

                test_fraction_done = (test_batchind+1) / test_num_batch
                avg_test_loss = avg_test_loss + loss.item()
                # print info and update log file
                log_string('[%s %d: %d/%d] %s loss: %f' % (opt.name, epoch, train_batchind,
                                                           train_num_batch-1, blue('test'), loss.item()), log_file)

                test_writer.add_scalar('total_loss', loss.item(),
                                       (epoch + test_fraction_done) * train_num_batch * opt.batchSize)
                test_writer.add_scalar('n_loss', n_loss.item(),
                                       (epoch + test_fraction_done) * train_num_batch * opt.batchSize)
                test_writer.add_scalar('err_angle', err_angle.item(),
                                       (epoch + test_fraction_done) * train_num_batch * opt.batchSize)
                test_writer.add_scalar('consistency_loss', consistency_loss.item(),
                                       (epoch + test_fraction_done) * train_num_batch * opt.batchSize)
                test_writer.add_scalar('normal_loss', normal_loss.item(),
                                       (epoch + test_fraction_done) * train_num_batch * opt.batchSize)
                test_writer.add_histogram('weights', weights.detach().cpu().numpy(),
                                           (epoch + test_fraction_done) * train_num_batch * opt.batchSize)
                test_writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                          (epoch + test_fraction_done) * train_num_batch * opt.batchSize)

        avg_test_loss = avg_test_loss / test_num_batch
        # update learning rate
        if opt.scheduler_type == 'step':
            scheduler.step()
        else:
            scheduler.step(avg_test_loss)
        test_writer.add_scalar('avg_loss', avg_test_loss,
                               (epoch + test_fraction_done) * train_num_batch * opt.batchSize)

        # save model, overwriting the old model
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch-1:
            log_string("saving model to file :{}".format(model_filename),log_file)
            torch.save(model.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 10 == 0 or epoch == opt.nepoch-1:
            log_string("saving model to file :{}".format('%s_model_%d.pth' % (opt.name, epoch)), log_file)
            torch.save(model.state_dict(), os.path.join(out_dir, '%s_model_%d.pth' % (opt.name, epoch)))


def compute_loss(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, normal_loss_type, arch,
                 patch_rot=None, phase='train',
                 use_consistency=False, point_weights=None, neighbor_normals=None, opt=None, trans=None, trans2=None):

    loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
    n_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
    consistency_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)

    # # generate a inv normal distribution for weight kl div
    # add pointnet transformation regularization
    regularizer_trans = 0
    if trans is not None:
        regularizer_trans += 0.1 * torch.nn.MSELoss()(trans * trans.permute(0, 2, 1),
                                                torch.eye(3, device=trans.device).unsqueeze(0).repeat(
                                                    trans.size(0), 1, 1))
    if trans2 is not None:
        regularizer_trans += 0.01 * torch.nn.MSELoss()(trans2 * trans2.permute(0, 2, 1),
                                                 torch.eye(64, device=trans.device).unsqueeze(0).repeat(
                                                     trans.size(0), 1, 1))

    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_target = target[output_target_ind[oi]]
            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

            if o == 'unoriented_normals':
                if normal_loss_type == 'ms_euclidean':
                    normal_loss = torch.min((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean() * output_loss_weight[oi]
                elif normal_loss_type == 'ms_oneminuscos':
                    cos_ang = normal_estimation_utils.cos_angle(o_pred, o_target)
                    normal_loss = (1-torch.abs(cos_ang)).pow(2).mean() * output_loss_weight[oi]
                elif normal_loss_type == 'sin':
                    normal_loss = 0.5 * torch.norm(torch.cross(o_pred, o_target, dim=-1), p=2, dim=1).mean()
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss_type))
                loss = normal_loss
                # get angle value at test time (not in training to save runtime)
                if phase == 'test':
                    if not normal_loss_type == 'ms_oneminuscos':
                        cos_ang = torch.abs(normal_estimation_utils.cos_angle(o_pred, o_target))
                        cos_ang[cos_ang>1] = 1
                    angle = torch.acos(cos_ang)
                    err_angle = torch.mean(angle)
                else:
                    err_angle = None
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        elif o == 'neighbor_normals':
            if use_consistency:
                o_pred = neighbor_normals
                o_target = target[output_target_ind[oi]]

                batch_size, n_points, _ = neighbor_normals.shape
                if patch_rot is not None:
                    # transform predictions with inverse transform
                    o_pred = torch.bmm(o_pred.view(-1, 1, 3),
                                       patch_rot.transpose(2, 1).repeat(1, n_points, 1, 1).view(-1, 3, 3)).view(batch_size, n_points, 3)
                # if opt.jet_order < 2: # when the jet has order higher than 2 the normal vector orientation matters.
                if normal_loss_type == 'ms_euclidean':
                    consistency_loss = torch.mean(point_weights * torch.min((o_pred - o_target).pow(2).sum(2),
                                                                            (o_pred + o_target).pow(2).sum(2)) )
                elif normal_loss_type == 'ms_oneminuscos':
                    cos_ang = normal_estimation_utils.cos_angle(o_pred.view(-1, 3),
                                                                o_target.view(-1, 3)).view(batch_size, n_points)
                    consistency_loss = torch.mean(point_weights * (1 - torch.abs(cos_ang)).pow(2))
                elif normal_loss_type == 'sin':
                    consistency_loss = 0.25 * torch.mean(point_weights *
                                                  torch.norm(torch.cross(o_pred.view(-1, 3),
                                                                         o_target.view(-1, 3), dim=-1).view(batch_size, -1, 3), p=2, dim=2))
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss_type))

                if opt.con_reg == 'mean':
                    regularizer = - 0.01 * torch.mean(point_weights)
                elif opt.con_reg == "log":
                    regularizer = - 0.05 * torch.mean(point_weights.log())
                elif opt.con_reg == 'norm':
                    regularizer = torch.mean((1/n_points)*torch.norm(point_weights-1, dim=1))
                else:
                    raise ValueError("Unkonwn consistency regularizer")
                regularizer = regularizer_trans + regularizer
                consistency_loss = consistency_loss + regularizer

                loss = consistency_loss + normal_loss
        else:
            raise ValueError('Unsupported output type: %s' % (o))

    return loss, n_loss, err_angle, consistency_loss, normal_loss


def get_data_loaders(opt, target_features):
    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=opt.neighbor_search)
    if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=opt.neighbor_search)

    if opt.training_order == 'random':
        test_datasampler = RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    return train_dataloader, train_dataset, train_datasampler, test_dataloader, test_dataset, test_datasampler


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

    return target_features, output_target_ind, output_pred_ind, output_loss_weight


def get_model(opt, log_dirname):
    # create model
    if opt.arch == 'simple':
        model = DeepFit.DeepFit(1, opt.points_per_patch,
                                            use_point_stn=opt.use_point_stn, use_feat_stn=opt.use_feat_stn,
                                            point_tuple=opt.point_tuple, sym_op=opt.sym_op,
                                            jet_order=opt.jet_order,
                                            weight_mode=opt.weight_mode, use_consistency=opt.use_consistency).cuda()
    elif opt.arch == '3dmfv':
        model = DeepFit.DeepFit(1, opt.points_per_patch,
                                            use_point_stn=opt.use_point_stn,
                                            use_feat_stn=opt.use_feat_stn, point_tuple=opt.point_tuple,
                                            sym_op=opt.sym_op, arch=opt.arch, n_gaussians=opt.n_gaussians,
                                            jet_order=opt.jet_order,
                                            weight_mode=opt.weight_mode, use_consistency=opt.use_consistency).cuda()
    else:
        raise ValueError('Unsupported architecture type')
    os.system('cp %s %s' % (
        './models/DeepFit.py', os.path.join(log_dirname, 'DeepFit.py')))  # backup the current model
    return model

if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)
