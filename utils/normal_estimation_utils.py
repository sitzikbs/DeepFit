# normal_estimation_utils.py normal estimation and 3DmFV utility functions
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import os
import pickle
import numpy as np
import torch

import provider


def get_gmm(points, n_gaussians, NUM_POINT, type='grid', variance=0.05, n_scales=3, D=3):
    """
    Compute weights, means and covariances for a gmm with two possible types 'grid' (2D/3D) and 'learned'
    Input:
         points: num_points_per_model*nummodels X 3 - xyz coordinates
         n_gaussians: scalar of number of gaussians /  number of subdivisions for grid type
         NUM_POINT: number of points per model
         type: 'grid' / 'leared' toggle between gmm methods
         variance: gaussian variance for grid type gmm
    Return:
         gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    if type == 'grid':
        #Generate gaussians on a grid - supports only axis equal subdivisions
        if n_gaussians >= 32:
            print('Warning: You have set a very large number of subdivisions.')
        if not(isinstance(n_gaussians, list)):
            if D == 2:
                gmm = get_2d_grid_gmm(subdivisions=[n_gaussians, n_gaussians], variance=variance)
            elif  D == 3 :
                gmm = get_3d_grid_gmm(subdivisions=[n_gaussians, n_gaussians, n_gaussians], variance=variance)
            else:
                ValueError('Wrong dimension. This supports either D=2 or D=3')

    elif type == 'learn':
        #learn the gmm from given data and save to gmm.p file, if already learned then load it from existing gmm.p file for speed
        if isinstance(n_gaussians, list):
            raise  ValueError('Wrong number of gaussians: non-grid value must be a scalar')
        print("Computing GMM from data - this may take a while...")
        info_str = "g" + str(n_gaussians) + "_N" + str(len(points)) + "_M" + str(len(points) / NUM_POINT)
        gmm_dir = "gmms"
        if not os.path.exists(gmm_dir): os.mkdir(gmm_dir)
        filename = gmm_dir + "/gmm_" + info_str + ".p"
        if os.path.isfile(filename):
            gmm = pickle.load(open(filename, "rb"))
        else:
            gmm = get_learned_gmm(points, n_gaussians, covariance_type='diag')
            pickle.dump(gmm, open( filename, "wb" ) )
    else:
        ValueError('Wrong type of GMM [grid/learn]')

    return gmm


def get_learned_gmm(points, n_gaussians, covariance_type='diag'):
    """
    Learn weights, means and covariances for a gmm based on input data
    Input:
         points: num_points_per_model*nummodels X 3 - xyz coordinates
         n_gaussians: scalar of number of gaussians /  3 element list of number of subdivisions for grid type
         covariance_type: Specify the type of covariance mmatrix : 'diag', 'full','tied', 'spherical' (Note that the Fisher Vector method relies on diagonal covariance matrix)
        See sklearn documentation : http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    Return:
         gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    gmm = GaussianMixture(n_components = n_gaussians, covariance_type=covariance_type)
    gmm.fit(points.astype(np.float64))
    return gmm


def get_3d_grid_gmm(subdivisions=[5,5,5], variance=0.04):
    """
    Compute the weight, mean and covariance of a gmm placed on a 3D grid
    Input:
        subdivisions: 2 element list of number of subdivisions of the 3D space in each axes to form the grid
        variance: scalar for spherical gmm.p
    Return:
         gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    # n_gaussians = reduce(lambda x, y: x*y,subdivisions)
    n_gaussians = np.prod(np.array(subdivisions))
    step = [ 1.0/(subdivisions[0]),  1.0/(subdivisions[1]),  1.0/(subdivisions[2])]

    means = np.mgrid[ step[0]-1 : 1.0-step[0]: complex(0,subdivisions[0]),
                      step[1]-1 : 1.0-step[1]: complex(0,subdivisions[1]),
                      step[2]-1 : 1.0-step[2]: complex(0,subdivisions[2])]
    means = np.reshape(means, [3,-1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm


def get_2d_grid_gmm(subdivisions=[5,5], variance=0.04):
    """
    Compute the weight, mean and covariance of a 2D gmm placed on a 2D grid
    Input:
        subdivisions: 2 element list of number of subdivisions of the 2D space in each axes to form the grid
        variance: scalar for spherical gmm.p
    Return:
         gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    # n_gaussians = reduce(lambda x, y: x*y,subdivisions)
    n_gaussians = np.prod(np.array(subdivisions))
    step = [ 1.0/(subdivisions[0]),  1.0/(subdivisions[1])]

    means = np.mgrid[ step[0]-1 : 1.0-step[0]: complex(0,subdivisions[0]),
                      step[1]-1 : 1.0-step[1]: complex(0,subdivisions[1])]
    means = np.reshape(means, [2,-1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm


def get_fisher_vectors(points,gmm, normalization=True):
    """
    :param points: n_points x 64 / B x n_points x 64
    :param gmm: sklearn MixtureModel class containing the gmm.p parameters.p
    :return: fisher vector representation for a single point cloud or a batch of point clouds
    """
    #fv = fisher_vector_par(points, gmm.p, normalization=True)


    if len(points.shape) == 2:
        #Single model
        fv = fisher_vector(points, gmm, normalization=normalization)
    else:
        # Batch of models
        fv = []
        n_models = points.shape[0]
        for i in range(n_models):
            fv.append(fisher_vector(points[i], gmm, normalization=True))
        fv = np.array(fv)
    return fv


def fisher_vector(xx, gmm, normalization=True):
    """Computes the Fisher vector on a set of descriptors.
    code from : https://gist.github.cnsom/danoneata/9927923
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 128 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    Sanchez, J., Perronnin, F., Mensink, T., & Verbeek, J. (2013).
    Image classification with the fisher vector: Theory and practice. International journal of computer vision, 105(64), 222-245.
    https://hal.inria.fr/hal-00830491/file/journal.pdf

    """
    xx = np.atleast_2d(xx)
    n_points = xx.shape[0]
    D = gmm.means_.shape[1]
    tiled_weights = np.tile(np.expand_dims(gmm.weights_, axis=-1), [1, D])

    #start = time.time()
    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK
    #mid = time.time()
    #print("Computing the probabilities took ", str(mid-start))
    #Compute Derivatives

    # Compute the sufficient statistics of descriptors.
    s0 = np.sum(Q, 0)[:, np.newaxis] / n_points
    s1 = np.dot(Q.T, xx) / n_points
    s2 = np.dot(Q.T, xx ** 2) / n_points

    d_pi = (s0.squeeze() - n_points * gmm.weights_) / np.sqrt(gmm.weights_)
    d_mu = (s1 - gmm.means_ * s0 ) / np.sqrt(tiled_weights*gmm.covariances_)
    d_sigma = (
        + s2
        - 2 * s1 * gmm.means_
        + s0 * gmm.means_ ** 2
        - s0 * gmm.covariances_
        ) / (np.sqrt(2*tiled_weights)*gmm.covariances_)

    #Power normaliation
    alpha = 0.5
    d_pi = np.sign(d_pi) * np.power(np.absolute(d_pi),alpha)
    d_mu = np.sign(d_mu) * np.power(np.absolute(d_mu), alpha)
    d_sigma = np.sign(d_sigma) * np.power(np.absolute(d_sigma), alpha)

    if normalization == True:
        d_pi = normalize(d_pi[:,np.newaxis], axis=0).ravel()
        d_mu = normalize(d_mu, axis=0)
        d_sigma = normalize(d_sigma, axis=0)
    # Merge derivatives into a vector.

    #print("comnputing the derivatives took ", str(time.time()-mid))

    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def fisher_vector_per_point( xx, gmm):
    """
    see notes for above function - performs operations per point
    INPUT:
    xx: array_like, shape (N, D) or (D, )- The set of descriptors

    gmm: instance of sklearn mixture.GMM object - Gauassian mixture model of the descriptors.
    OUTPUT:
    fv_per_point : fisher vector per point (derivative by w, derivative by mu, derivative by sigma)
    """
    xx = np.atleast_2d(xx)
    n_points = xx.shape[0]
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]

    sig2 = np.array([gmm.covariances_.T[0,:],gmm.covariances_.T[1,:],gmm.covariances_.T[2,:]]).T
    sig2_tiled = np.tile(np.expand_dims(sig2,axis=0),[n_points,1,1])

    # Compute derivativees per point and then sum.
    Q = gmm.predict_proba(xx)  # NxK
    tiled_weights = np.tile(np.expand_dims(gmm.weights_, axis=-1), [1, D])
    sqrt_w = np.sqrt(tiled_weights)

    d_pi = (Q - np.tile(np.expand_dims(gmm.weights_,0),[n_points,1])) / np.sqrt(np.tile(np.expand_dims(gmm.weights_,0),[n_points,1]))
    x_mu = np.tile( np.expand_dims(xx,axis=2),[1,1, n_gaussians]) - np.tile(np.expand_dims(gmm.means_.T,axis=0),[n_points,1,1])
    x_mu = np.swapaxes(x_mu,1,2)
    d_mu = (np.tile(np.expand_dims(Q,-1),D) * x_mu) / (np.sqrt(sig2_tiled) * sqrt_w)

    d_sigma =   np.tile(np.expand_dims(Q, -1), 3)*((np.power(x_mu,2)/sig2_tiled)-1)/(np.sqrt(2)*sqrt_w)

    fv_per_point = (d_pi, d_mu, d_sigma)
    return fv_per_point


def l2_normalize(v, dim=1):
    #normalize a vector along a dimension
    #INPUT:
    # v: a vector or matrix to normalize
    # dim : the dimension along which to normalize
    #OUTPUT: normalized v along dim
    norm = np.linalg.norm(v, axis=dim)
    if norm.all() == 0:
       return v
    return v / norm


def get_3DmFV(points, w, mu, sigma, normalize=True):
    """
       Compute the 3D modified fisher vectors given the gmm model parameters (w,mu,sigma) and a set of points
       For faster performance (large batches) use the tensorflow version

       :param points: B X N x 3 tensor of XYZ points
       :param w: B X n_gaussians tensor of gaussian weights
       :param mu: B X n_gaussians X 3 tensor of gaussian cetnters
       :param sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
       :return: fv: B X 20*n_gaussians tensor of the fisher vector
       """
    n_batches = points.shape[0]
    n_points = points.shape[1]
    n_gaussians = mu.shape[0]
    D = mu.shape[1]

    # Expand dimension for batch compatibility
    batch_sig = np.tile(np.expand_dims(sigma, 0), [n_points, 1, 1])  # n_points X n_gaussians X D
    batch_sig = np.tile(np.expand_dims(batch_sig, 0), [n_batches, 1, 1, 1])  # n_batches X n_points X n_gaussians X D
    batch_mu = np.tile(np.expand_dims(mu, 0), [n_points, 1, 1])  # n_points X n_gaussians X D
    batch_mu = np.tile(np.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1])  # n_batches X n_points X n_gaussians X D
    batch_w = np.tile(np.expand_dims(np.expand_dims(w, 0), 0), [n_batches, n_points,
                                                                1])  # n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = np.tile(np.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1])  # n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    # Compute derivatives
    # w_per_batch = tf.tile(tf.expand_dims(w,0),[n_batches, 1]) #n_batches X n_gaussians
    w_per_batch_per_d = np.tile(np.expand_dims(np.expand_dims(w, 0), -1),
                                [n_batches, 1, 3*D])  # n_batches X n_gaussians X 128*D (D for min and D for max)

    # Define multivariate noraml distributions
    # Compute probability per point
    p_per_point = (1.0 / (np.power(2.0 * np.pi, D / 2.0) * np.power(batch_sig[:, :, :, 0], D))) * np.exp(
        -0.5 * np.sum(np.square((batch_points - batch_mu) / batch_sig), axis=3))

    w_p = p_per_point
    Q = w_p  # enforcing the assumption that the sum is 1
    Q_per_d = np.tile(np.expand_dims(Q, -1), [1, 1, 1, D])

    d_pi_all = np.expand_dims((Q - batch_w) / (np.sqrt(batch_w)), -1)
    d_pi = np.concatenate([np.max(d_pi_all, axis=1), np.sum(d_pi_all, axis=1)], axis=2)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1 / (np.sqrt(w_per_batch_per_d))) * np.concatenate([np.max(d_mu_all, axis=1), np.min(d_mu_all, axis=1), np.sum(d_mu_all, axis=1)], axis=2)

    d_sig_all = Q_per_d * (np.square((batch_points - batch_mu) / batch_sig) - 1)
    d_sigma = (1 / (np.sqrt(2 * w_per_batch_per_d))) * np.concatenate([np.max(d_sig_all, axis=1), np.min(d_sig_all, axis=1), np.sum(d_sig_all, axis=1)], axis=2)

    # number of points  normaliation
    d_pi = d_pi / n_points
    d_mu = d_mu / n_points
    d_sigma =d_sigma / n_points

    if normalize:
        # Power normaliation
        alpha = 0.5
        d_pi = np.sign(d_pi) * np.power(np.abs(d_pi), alpha)
        d_mu = np.sign(d_mu) * np.power(np.abs(d_mu), alpha)
        d_sigma = np.sign(d_sigma) * np.power(np.abs(d_sigma), alpha)

        # L2 normaliation
        d_pi = np.array([l2_normalize(d_pi[i,:, :], dim=0) for i in range (n_batches)])
        d_mu = np.array([l2_normalize(d_mu[i, :,:], dim=0) for i in range (n_batches)])
        d_sigma = np.array([l2_normalize(d_sigma[i, :, :], dim=0) for i in range (n_batches)])


    fv = np.concatenate([d_pi, d_mu, d_sigma], axis=2)
    fv = np.transpose(fv, axes=[0, 2, 1])

    return fv

def euclidean_to_spherical(points, format='degrees'):
    """
    euclidean_to_spherical converts a point from its xyz coordinates to phi, teta, coordinates.
    It assumes and enforces r=1.
    It also assumes :
    x = r * sin(teta) * cos(phi)
    y = r * sin(teta) * sin(phi)
    z = r * cos(teta)

    :param points: B x 3 xyz coordinates
    :return: teta: B x 1 angle between z axis to the vector (ISO convention)
    :return phi : B x 1 angle between the x axis and the projection of the vector on the xy plain (ISO convention)
    """
    # r = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    teta = np.arctan2(np.sqrt(points[:, 0]**2 + points[:, 1]**2), points[:, 2])
    phi = np.arctan2(points[:, 1], points[:, 0])

    if format == 'degrees':
        phi = np.rad2deg(phi)
        teta = np.rad2deg(teta)

    return phi, teta

def cos_angle(v1, v2):

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out

def compute_principal_curvatures(beta):
    """
    given the jet coefficients, compute the principal curvatures and principal directions:
    the eigenvalues and eigenvectors of the weingarten matrix
    :param beta: batch of Jet coefficients vector
    :return: k1, k2, dir1, dir2: batch of principal curvatures and principal directions
    """
    with torch.no_grad():
        if beta.shape[1] < 5:
            raise ValueError("Can't compute curvatures for jet with order less than 2")
        else:
            b1_2 = torch.pow(beta[:, 0], 2)
            b2_2 = torch.pow(beta[:, 1], 2)
            #first fundemental form
            E = (1 + b1_2).view(-1, 1, 1)
            G = (1 + b2_2).view(-1, 1, 1)
            F = (beta[:, 1] * beta[:, 0]).view(-1, 1, 1)
            I = torch.cat([torch.cat([E, F], dim=2), torch.cat([F, G], dim=2)], dim=1)
            # second fundemental form
            norm_N0 = torch.sqrt(b1_2 + b2_2 + 1)
            e = (2*beta[:, 2] / norm_N0).view(-1, 1, 1)
            f = (beta[:, 4] / norm_N0).view(-1, 1, 1)
            g = (2*beta[:, 3] / norm_N0).view(-1, 1, 1)
            II = torch.cat([torch.cat([e, f], dim=2), torch.cat([f, g], dim=2)], dim=1)

            M_weingarten = -torch.bmm(torch.inverse(I), II)

            curvatures, dirs = torch.symeig(M_weingarten, eigenvectors=True) #faster
            dirs = torch.cat([dirs, torch.zeros(dirs.shape[0], 2, 1, device=dirs.device)], dim=2) # pad zero in the normal direction

    return curvatures, dirs


if __name__ == "__main__":

    model_idx = 0
    num_points = 1024
    gmm = get_3d_grid_gmm(subdivisions=[5, 5, 5], variance=0.04)
    points = provider.load_single_model(model_idx=model_idx, train_file_idxs=0, num_points=num_points)
    points = np.tile(np.expand_dims(points,0),[128,1,1])

    fv_gpu = get_fisher_vectors(points,gmm, normalization=True)
