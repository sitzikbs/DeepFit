import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, './utils'))
from pcpnet_dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler,\
    SequentialPointcloudPatchSampler

from sklearn.neighbors import KDTree
import random

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def translate_point_cloud(batch_data, tval = 0.2):
    """ Randomly translate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, translated batch of point clouds
    """
    n_batches = batch_data.shape[0]
    n_points = batch_data.shape[1]
    translation = np.random.uniform(-tval, tval, size=[n_batches,3])
    translation = np.tile(np.expand_dims(translation,1),[1,n_points,1])
    batch_data = batch_data + translation
    # for k in xrange(n_batches):
    #     batch_data[k, ...] = batch_data[k, ...] + translation[k]
    return batch_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 128 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_x_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along x direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 128 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def scale_point_cloud(batch_data, smin = 0.66, smax = 1.5):
    """ Randomly scale the point clouds to augument the dataset
        scale is per shape
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, scaled batch of point clouds
    """
    scaled = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        sx = np.random.uniform(smin, smax)
        sy = np.random.uniform(smin, smax)
        sz = np.random.uniform(smin, smax)
        scale_matrix = np.array([[sx, 0, 0],
                                    [0, sy, 0],
                                    [0, 0, sz]])
        shape_pc = batch_data[k, ...]
        scaled[k, ...] = np.dot(shape_pc.reshape((-1, 3)), scale_matrix)
    return scaled


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def insert_outliers_to_point_cloud(batch_data, outlier_ratio=0.05):
    """ inserts log_noise Randomly distributed in the unit sphere
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array,  batch of point clouds with log_noise
    """
    B, N, C = batch_data.shape
    outliers = np.random.uniform(-1, 1, [B, int(np.floor(outlier_ratio * N)), C])
    points_idx = np.random.choice(range(0, N), int(np.ceil(N * (1 - outlier_ratio))))
    outlier_data = np.concatenate([batch_data[:, points_idx, :], outliers], axis=1)
    return outlier_data


def occlude_point_cloud(batch_data, occlusion_ratio):
    """ Randomly k remove points (number of points defined by the ratio.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          Bx(N-k)x3 array, occluded batch of point clouds
    """
    B, N, C = batch_data.shape
    k = int(np.round(N*occlusion_ratio))
    occluded_batch_point_cloud = []
    for i in range(B):
        point_cloud = batch_data[i, :, :]
        kdt = KDTree(point_cloud, leaf_size=30, metric='euclidean')
        center_of_occlusion = random.choice(point_cloud)
        #occluded_points_idx = kdt.query_radius(center_of_occlusion.reshape(1, -1), r=occlusion_radius)
        _, occluded_points_idx = kdt.query(center_of_occlusion.reshape(1, -1), k=k)
        point_cloud = np.delete(point_cloud, occluded_points_idx, axis=0)
        occluded_batch_point_cloud.append(point_cloud)
    return np.array(occluded_batch_point_cloud)



def starve_gaussians(batch_data, gmm, starv_coef=0.6, n_points=1024):
    """ sample points from a point cloud with specific sparse regions (defined by the gmm gaussians)
        Input:
          batch_data: BxNx3 array, original batch of point clouds
          gmm: gausian mixture model
        Return:
          BxNx3 array, jittered batch of point clouds
    """

    B, N, D = batch_data.shape
    n_gaussians = len(gmm.weights_)
    choices = [1, starv_coef]
    mu = gmm.means_
    #find a gaussian for each point
    mu = np.tile(np.expand_dims(np.expand_dims(mu,0),0),[B,N,1,1]) #B X N X n_gaussians X D
    batch_data_per_gaussian = np.tile(np.expand_dims(batch_data,-2),[1, 1, n_gaussians, 1] )
    d = np.sum(np.power(batch_data_per_gaussian-mu,2), -1)
    idx = np.argmin(d, axis=2)

    #compute servival probability
    rx = np.random.rand(B, N)
    sk = np.random.choice(choices, n_gaussians)
    p = sk[idx] * rx
    starved_points = []
    for i in range(B):
        topmostidx = np.argsort(p[i,:])[::-1][:n_points]
        starved_points.append(batch_data[i,topmostidx,:])
    return np.asarray(starved_points)


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def replace_labels(numbers, problem_numbers, alternative_numbers):
    # Replace values
    problem_numbers = np.asarray(problem_numbers)
    alternative_numbers = np.asarray(alternative_numbers)
    n_min, n_max = numbers.min(), numbers.max()
    replacer = np.arange(n_min, n_max + 1)
    mask = problem_numbers <= n_max  # Discard replacements out of range
    replacer[problem_numbers[mask] - n_min] = alternative_numbers[mask]
    numbers = replacer[numbers - n_min]
    return numbers

def get_data_loader(dataset_name='trainingset_temp.txt', batchSize=128, indir='./pclouds', patch_radius=[0.05],
                          points_per_patch=500, outputs=['unoriented_normals'], patch_point_count_std=0,
                          seed=3627473, identical_epochs=False, use_pca=False, patch_center='point',
                          point_tuple=1, cache_capacity=100, patches_per_shape=1000, patch_sample_order='random',
                          workers=0, dataset_type='training', sparse_patches=False):
    """
    Helper function to load the pcpnet datasets using their dataloader class

    :param indir: input folder (point clouds)
    :param dataset_name: dataset file list name
    :param patch_radius: patch radius in multiples of the shape's bounding box diagonal, multiple values for multi-scale.
    :param points_per_patch:  number of points per patch
    :param outputs: outputs of the network, a list with elements of:
                            unoriented_normals: unoriented (flip-invariant) point normals
                             oriented_normals: oriented point normals
                             max_curvature: maximum curvature
                             min_curvature: mininum curvature
    :param patch_point_count_std: standard deviation of the number of points in a patch
    :param seed: manual seed
    :param identical_epochs: use same patches in each epoch, mainly for debugging
    :param use_pca: Give both inputs and ground truth in local PCA coordinate frame
    :param patch_center: center patch at - 'point': center point / 'mean': patch mean
    :param point_tuple: use n-tuples of points as input instead of single points
    :param cache_capacity: Max. number of dataset elements (usually shapes) to hold in the cache at the same time.
    :param patches_per_shape: number of patches sampled from each shape in an epoch
    :param patch_sample_order: order in which the training patches are presented:
                            'full': evaluate all points in the dataset
                            'random': fully random over the entire dataset (the set of all patches is permuted)
                             'random_shape_consecutive': random over the entire dataset, but patches of a shape remain
                             consecutive (shapes and patches inside a shape are permuted)

    :param batchSize: input batch size
    :param workers: number of data loading workers - 0 means same thread as main execution
    :param dataset_type: 'training' / 'validation' / 'test' - used only for printing
    :param sparse_patches: evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.
    :return: dataloader: pcpnet data loader object
    """

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in outputs:
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
        elif o == 'noise':
            target_features.append(o)
            pred_dim += 1
        else:
            raise ValueError('Unknown output: %s' % (o))

    dataset = PointcloudPatchDataset(
        root=indir,
        shape_list_filename=dataset_name,
        patch_radius=patch_radius,
        points_per_patch=points_per_patch,
        patch_features=target_features,
        point_count_std=patch_point_count_std,
        seed=seed,
        identical_epochs=identical_epochs,
        use_pca=use_pca,
        center=patch_center,
        point_tuple=point_tuple,
        cache_capacity=cache_capacity,
        sparse_patches=sparse_patches)

    if patch_sample_order == 'random':
        datasampler = RandomPointcloudPatchSampler(
            dataset,
            patches_per_shape=patches_per_shape,
            seed=seed,
            identical_epochs=identical_epochs)
    elif patch_sample_order == 'random_shape_consecutive':
        datasampler = SequentialShapeRandomPointcloudPatchSampler(
            dataset,
            patches_per_shape=patches_per_shape,
            seed=seed,
            identical_epochs=identical_epochs)
    elif patch_sample_order == 'full':
        datasampler = SequentialPointcloudPatchSampler(dataset)
    else:
        raise ValueError('Unknown patch sampling order: %s' % (training_order))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=batchSize,
        num_workers=int(workers))

    print(dataset_type + ' set: %d patches (in %d batches))' %
          (len(datasampler), len(dataloader)))
    return (dataloader, dataset)


if __name__ == '__main__':
    pc_path = '/home/sitzikbs/PycharmProjects/pcpnet/pclouds/'
    testset_filename = pc_path + 'testset.txt'
    validationset_filename = pc_path + 'validationset.txt'
    trainset_filename = pc_path + 'trainingset.txt'

    train_data_loader = get_data_loader(dataset_name=trainset_filename, batchSize=128, indir=pc_path,
                                        patch_radius=[0.05], points_per_patch=500, outputs=['unoriented_normals'],
                                        patch_point_count_std=0, seed=3627473, identical_epochs=False, use_pca=False,
                                        patch_center='point', point_tuple=1, cache_capacity=100, patches_per_shape=1000,
                                        patch_sample_order='random', workers=0, dataset_type='training')
    validation_data_oader = get_data_loader(dataset_name=validationset_filename, batchSize=128, indir=pc_path,
                                        patch_radius=[0.05], points_per_patch=500, outputs=['unoriented_normals'],
                                        patch_point_count_std=0, seed=3627473, identical_epochs=False, use_pca=False,
                                        patch_center='point', point_tuple=1, cache_capacity=100, patches_per_shape=1000,
                                            patch_sample_order='random', workers=0, dataset_type='validation')
    test_data_loader = get_data_loader(dataset_name=testset_filename, batchSize=128, indir=pc_path,
                                        patch_radius=[0.05], points_per_patch=500, outputs=['unoriented_normals'],
                                        patch_point_count_std=0, seed=3627473, identical_epochs=False, use_pca=False,
                                        patch_center='point', point_tuple=1, cache_capacity=100, patches_per_shape=1000,
                                       patch_sample_order='full', workers=0, dataset_type='test')
    print(1)