import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


def fit_Wjet(points, weights, order=2, compute_neighbor_normals=False, compute_residuals=False):
    """
    Fit a "jet" - n-order polynomial to a point clouds with weighted ponits. assumes pca was performed on the points beforehand
    :param points: xyz ponits
    :param W: weight vector (per point)
    :param order: n-order of the polynomial
    :return: beta: polynomial coefficients
    :return: n_est: normal estimation

    """
    neighbor_normals = None
    residuals = None
    batch_size, D, n_points = points.shape
    # compute the vandermonde matrix
    x = points[:, 0, :].unsqueeze(-1)
    y = points[:, 1, :].unsqueeze(-1)
    z = points[:, 2, :].unsqueeze(-1)
    weights = weights.unsqueeze(-1)
    # handle zero weights - if all weights are zero set them to 1
    valid_count = torch.sum(weights > 1e-3, dim=1).unsqueeze(-1)
    W_learned = torch.diag_embed(weights.view(batch_size, -1))
    W_ones = torch.diag_embed(torch.ones_like(weights, requires_grad=True).view(batch_size, -1))
    W = torch.where(valid_count > 18, W_learned, W_ones)


    if order > 1:
        #pre conditioning
        h = (torch.mean(torch.abs(x), 1) + torch.mean(torch.abs(y), 1)) / 2 # absolute value added from https://github.com/CGAL/cgal/blob/b9e320659e41c255d82642d03739150779f19575/Jet_fitting_3/include/CGAL/Monge_via_jet_fitting.h
        # h = torch.mean(torch.sqrt(x*x + y*y), dim=2)
        idx = torch.abs(h) < 0.0001
        h[idx] = 0.1
        # h = 0.1 * torch.ones(batch_size, 1, device=points.device)
        x = x / h.unsqueeze(-1).repeat(1, n_points, 1)
        y = y / h.unsqueeze(-1).repeat(1, n_points, 1)

    if order == 1:
        A = torch.cat([x, y, torch.ones_like(x)], dim=2)
    elif order == 2:
        A = torch.cat([x, y, x * x, y * y, x * y, torch.ones_like(x)], dim=2)
        h_2 = h * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, torch.ones_like(h)], dim=1))
    elif order == 3:
        y_2 = y * y
        x_2 = x * x
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_2 * x, y_2 * y, x_2 * y, y_2 * x,  torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, torch.ones_like(h)], dim=1))
    elif order == 4:
        y_2 = y * y
        x_2 = x * x
        x_3 = x_2 * x
        y_3 = y_2 * y
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2,
                       torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        h_4 = h_3 * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, h_4, h_4, h_4, h_4, h_4, torch.ones_like(h)], dim=1))
    else:
        raise ValueError("Polynomial order unsupported, please use 1 or 2 ")


    XtX = torch.matmul(A.permute(0, 2, 1),  torch.matmul(W, A))
    XtY = torch.matmul(A.permute(0, 2, 1), torch.matmul(W, z))

    beta = solve_linear_system(XtX, XtY, sub_batch_size=16)

    if order > 1: #remove preconditioning
         beta = torch.matmul(D_inv, beta)

    n_est = torch.nn.functional.normalize(torch.cat([-beta[:, 0:2].squeeze(-1), torch.ones(batch_size, 1, device=x.device)], dim=1), p=2, dim=1)

    if compute_neighbor_normals:
        beta_ = beta.squeeze().unsqueeze(1).repeat(1, n_points, 1).unsqueeze(-1)
        if order == 1:
            neighbor_normals = n_est.unsqueeze(1).repeat(1, n_points, 1)
        elif order == 2:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)
        elif order == 3:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y + 3 * beta_[:, :, 5] *  x_2 +
                             2 *beta_[:, :, 7] * xy + beta_[:, :, 8] * y_2),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x + 3 * beta_[:, :, 6] * y_2 +
                             beta_[:, :, 7] * x_2 + 2 * beta_[:, :, 8] * xy),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)
        elif order == 4:
            # [x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y + 3 * beta_[:, :, 5] * x_2 +
                             2 * beta_[:, :, 7] * xy + beta_[:, :, 8] * y_2 + 4 * beta_[:, :, 9] * x_3 + 3 * beta_[:, :, 11] * x_2 * y
                             + beta_[:, :, 12] * y_3 + 2 * beta_[:, :, 13] * y_2 * x),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x + 3 * beta_[:, :, 6] * y_2 +
                             beta_[:, :, 7] * x_2 + 2 * beta_[:, :, 8] * xy + 4 * beta_[:, :, 10] * y_3 + beta_[:, :, 11] * x_3 +
                             3 * beta_[:, :, 12] * x * y_2 + 2 * beta_[:, :, 13] * y * x_2),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)
        if compute_residuals:
            residuals = torch.pow(torch.matmul(A, beta) - z, 2).squeeze(-1)

    return beta.squeeze(), n_est, neighbor_normals, residuals


def solve_linear_system(XtX, XtY, sub_batch_size=None):
    """
    solve linear system of equations. use sub batches to avoid MAGMA bug
    :param XtX: matrix of the coefficients
    :param XtY: vector of the
    :param sub_batch_size: size of mini mini batch to avoid MAGMA error, if None - uses the entire batch
    :return:
    """
    if sub_batch_size is None:
        sub_batch_size = XtX.size(0)
    n_iterations = int(XtX.size(0) / sub_batch_size)
    assert sub_batch_size%sub_batch_size == 0, "batch size should be a factor of {}".format(sub_batch_size)
    beta = torch.zeros_like(XtY)
    n_elements = XtX.shape[2]
    for i in range(n_iterations):
        try:
            L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)
            beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
        except:
            # # add noise to diagonal for cases where XtX is low rank
            eps = torch.normal(torch.zeros(sub_batch_size, n_elements, device=XtX.device),
                               0.01 * torch.ones(sub_batch_size, n_elements, device=XtX.device))
            eps = torch.diag_embed(torch.abs(eps))
            XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] + \
                eps * XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...]
            try:
                L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                    torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
            except:
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...], _ =\
                    torch.solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...])
    return beta

def fit_jet_batch(points, order=2):
    """
    Fit a "jet" - n-order polynomial to a set of point clouds. assumes pca was performed on the points beforehand
    :param points: xyz ponits
    :param W: wight matrix
    :param order: n-order of the polinomial
    :return: beta coefficients

    """
    batch_size, D, n_points = points.shape
    # compute the vandermonde matrix
    x = points[:, 0, :].unsqueeze(-1)
    y = points[:, 1, :].unsqueeze(-1)
    z = points[:, 2, :].unsqueeze(-1)

    if order > 1:
        #pre conditioning
        # h = (torch.mean(torch.abs(x), 1) + torch.mean(torch.abs(y), 1) ) / 2 # absolute balue added from https://github.com/CGAL/cgal/blob/b9e320659e41c255d82642d03739150779f19575/Jet_fitting_3/include/CGAL/Monge_via_jet_fitting.h
        h = torch.mean(torch.sqrt(torch.pow(x, 2) + torch.pow(y), 2)) # from the paper
        idx = torch.abs(h) < 0.0001
        h[idx] = 1
        x = x / h.unsqueeze(-1).repeat(1, n_points, 1)
        y = y / h.unsqueeze(-1).repeat(1, n_points, 1)

    if order == 1:
        A = torch.cat([x, y, torch.ones_like(x)], dim=2)
    elif order == 2:
        A = torch.cat([x, y, x * x, y * y, x * y, torch.ones_like(x)], dim=2)
        h_2 = h * h
        D = torch.diag_embed(torch.cat([h, h, h_2, h_2, h_2, torch.ones_like(h)], dim=1))
    elif order == 3:
        y_2 = y * y
        x_2 = x * x
        A = torch.cat([x, y, x_2, y_2, x * y, x_2 * x, y_2 * y, x_2 * y, y_2 * x,  torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        D = torch.diag_embed(torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, torch.ones_like(h)], dim=1))
    elif order == 4:
        y_2 = y * y
        x_2 = x * x
        x_3 = x_2 * x
        y_3 = y_2 * y
        A = torch.cat([x, y, x_2, y_2, x * y, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2,
                       torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        h_4 = h_3 * h
        D = torch.diag_embed(torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, h_4, h_4, h_4, h_4, h_4, torch.ones_like(h)], dim=1))
    else:
        raise ValueError("Polynomial order unsupported, please use 1 or 2 ")

    n_elements = A.shape[2]

    XtX = torch.matmul(A.permute(0, 2, 1),  A)
    XtY = torch.matmul(A.permute(0, 2, 1), z)
    try:
        LU, LU_pivots  = torch.lu(XtX) #should check if this is needed
    except:
        # add noise to diagonal for cases where XtX is low rank
        eps = torch.normal(torch.zeros(batch_size, n_elements, device=x.device), 0.01 * torch.ones(batch_size, n_elements, device=x.device))
        eps = torch.diag_embed(eps)
        XtX = XtX + eps * XtX
        LU, LU_pivots = torch.lu(XtX)

    beta = torch.lu_solve(XtY, LU, LU_pivots)

    if order > 1:
         beta = torch.matmul(D, beta)
    n_est = torch.nn.functional.normalize(torch.cat([-beta[:, 0:2].squeeze(), torch.ones(batch_size, 1, device=x.device)], dim=1), p=2, dim=1)

    return beta.squeeze(), n_est



