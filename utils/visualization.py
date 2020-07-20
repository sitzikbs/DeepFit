# visualization.py utility visualization functions.
# For normal vector visualization use MATLAB code
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import numpy as np
import torch
import os
# from plane_dataset import PlaneSet as Dataset
from matplotlib import rc
rc('font', **{'family':'DejaVu Sans Mono'})
import matplotlib.pyplot as plt
from matplotlib import cm

# rc('text', usetex=True)

def visualize_3d_points(points, xlimit=[-1, 1], ylimit=[-1, 1], zlimit=[-1, 1], ax=None, display=False, weights=None,
                        vistype='scale', img_name='default_name', export=False, c_range=[0, 2], scale= 30):
    '''
    displays a scatter plot of 3d points
    '''
    if weights is not None:
        if vistype=='scale':
            weights[weights < 1e-3] = 0.1
            weights = weights * 50
    else:
        weights = np.ones_like(points[0])

    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if vistype == 'scale':
        plt.scatter(x, y, zs=z, s=weights)
    else:
        cmap = cm.get_cmap('jet', 256)
        plt.scatter(x, y, zs=z, s=scale, c=weights, cmap=cmap)
        plt.clim(c_range[0], c_range[1])
        cbar = plt.colorbar()
        cbar.set_label('point weight')

    ax = plt.gca()
    rng = 1
    ax.auto_scale_xyz([-rng, rng], [-rng, rng], [-rng, rng])
    ax.set_xlim(xlimit[0], xlimit[1])
    ax.set_ylim(ylimit[0], ylimit[1])
    ax.set_zlim(zlimit[0], zlimit[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()
    ax.view_init(32.64, 45)


    if display:
        plt.show()
    if export:
        plt.savefig(img_name, bbox_inches='tight')
    # plt.close()

    return ax


def plot_parametric_plane(beta, color='r', ax=None, display=False, label_txt='', export=False, img_name='default_name',
                          alpha=1.0, show_eq=False):
    res = 30
    x = np.outer(np.linspace(-1, 1, res), np.ones(res))
    y = x.copy().T  # transpose
    z = beta[0] * x + beta[1] * y + beta[2]
    surf = ax.plot_surface(x, y, z, edgecolor='None', color=color,
                label=label_txt + ' z={:.2f}x + {:.2f}y + {:.2f}'.format(beta[0], beta[1], beta[2]), alpha=alpha)
    surf._facecolors2d = surf._facecolors3d
    surf._edgecolors2d = surf._edgecolors3d
    if show_eq:
        ax.legend()

    if display:
        plt.show()
    if export:
        plt.savefig(img_name)
    return ax


def plot_parametric_jet(beta, color='r', ax=None, display=False, label_txt='', export=False, img_name='default_name',
                        alpha=1.0, trans=None, surf_range=[-1, 1], show_eq=False):

    # find the order of the jet
    order = int(np.max(np.roots([1, 3, 2-2*len(beta)])))
    # generate grid
    res = 30
    x = np.outer(np.linspace(surf_range[0], surf_range[1], res), np.ones(res))
    y = x.copy().T  # transpose
    # generate the height map from the grid (improve using vandermonde matrix
    if order == 1:
        z = beta[0] * x + beta[1] * y + beta[2]
        jet_eq_str = 'z={:.2f}x + {:.2f}y + {:.2f}'.format(beta[0], beta[1], beta[2])
    elif order == 2:
        z = beta[0] * x + beta[1] * y + beta[2] * x*x + beta[3]*y*y  + beta[4]*x*y + beta[5]
        jet_eq_str = 'z={:.2f}x + {:.2f}y + {:.2f}x^2 + {:.2f}y^2 +  {:.2f}x*y + {:.2f}'.format(
            beta[0], beta[1], beta[2], beta[3], beta[4], beta[5])
    elif order == 3:
        x_2 = x*x
        y_2 = y*y
        z = beta[0] * x + beta[1] * y + beta[2] * x_2 + beta[3] * y_2 + beta[4] * x * y + \
            beta[5] * x_2*x + beta[6]*y*y*y + beta[7]*x_2*y + beta[8]*x*y_2 + beta[9]
        jet_eq_str = 'z={:.2f}x + {:.2f}y + {:.2f}x^2 + {:.2f}y^2 +  {:.2f}x*y + ...'.format(
            beta[0], beta[1], beta[2], beta[3], beta[4])
    elif order == 4:
        x_2 = x*x
        y_2 = y*y
        y_3 = y_2*y
        x_3 = x_2*x
        z = beta[0] * x + beta[1] * y + beta[2] * x_2 + beta[3] * y_2 + beta[4] * x * y + \
            beta[5] * x_3 + beta[6]*y_3 + beta[7]*x_2*y + beta[8]*x*y_2 + beta[9] * x_3*x + beta[10] * y_3*y +\
            beta[11]*x_3*y + beta[12]*x*y_3 + beta[13]*x_2*y_2 + beta[14]
        jet_eq_str = 'z={:.2f}x + {:.2f}y + {:.2f}x^2 + {:.2f}y^2 +  {:.2f}x*y + ...'.format(
            beta[0], beta[1], beta[2], beta[3], beta[4])
    else:
        raise ValueError("Unsupported Jet order")
    if not trans is None:
        points = np.array([x.flatten(), y.flatten(), z.flatten()]).transpose()
        points = np.dot(points, trans).transpose()
        x, y, z = points[0].reshape(res, res), points[1].reshape(res, res), points[2].reshape(res, res)
    surf = ax.plot_surface(x, y, z, edgecolor='None', color=color,
                label=label_txt + jet_eq_str, alpha=alpha)
    surf._facecolors2d = surf._facecolors3d
    surf._edgecolors2d = surf._edgecolors3d
    if show_eq:
        ax.legend()

    if display:
        plt.show()
    if export:
        plt.savefig(img_name)
    # plt.close()
    return ax

def plot_plane_normal(beta, points, color='r', ax=None, display=False, mode='all', export=False, img_name='default_img_name'):
        normal = beta[0:2].tolist()
        normal.append(-1.0)
        normal = normal / np.sqrt(np.sum(np.square(normal)))
        normals = np.expand_dims(normal, 0).repeat(len(points[0]), axis=0).T
        if mode == 'all':
            ax.quiver(points[0,:], points[1, :], points[2, :], -normals[0, :], -normals[1, :], -normals[2, :], length=0.2,
                      normalize=True, color=color)
        else:
            ax.quiver(points[0,0], points[1, 0], points[2, 0], -normals[0, 0], -normals[1, 0], -normals[2, 0], length=0.2,
                      normalize=True, color=color)

        if display:
            plt.show()
        if export:
            plt.savefig(img_name)


def plot_normals(normals, points, color='r', ax=None, display=False, mode='all', export=False, img_name='default_img_name'):
    if mode == 'all':
        ax.quiver(points[0, :], points[1, :], points[2, :], normals[0, :], normals[1, :], normals[2, :], length=0.4,
                  normalize=True, color=color, linewidth=3)
    else:
        ax.quiver(points[0, 0], points[1, 0], points[2, 0], normals[0, 0], normals[1, 0], normals[2, 0], length=0.4,
                  normalize=True, color=color, linewidth=3)
    if display:
        plt.show()
    if export:
        plt.savefig(img_name)
        plt.close()


def export_four_views(ax, image_name='default_name'):
    ax.set_axis_off()
    ax.view_init(32.64, 45)
    plt.draw()
    plt.savefig(image_name + '_iso.png')
    ax.view_init(0, 0)
    plt.draw()
    plt.savefig(image_name + '_side.png')
    ax.view_init(90, 0)
    plt.draw()
    plt.savefig(image_name + '_top.png')
    plt.close()


if __name__ == "__main__":

    batch_size = 8
    types = [x[0] for x in os.walk('./data/')]
    type = './data/corners'

    val_dataset = Dataset(root=type, train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                                 pin_memory=True)

    for i, data in enumerate(val_dataloader):
        index = 0
        points, beta = (data[0], data[1])
        ax = visualize_3d_points(points[index], display=False)
        plot_parametric_plane(beta[index].numpy(), color='r', ax=ax, display=True)

        # x = points[index, 0, :].numpy()
        # y = points[index, 1, :].numpy()
        # A = np.vstack([x, np.ones(len(x))]).T
        # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # plot_parametric_line(m, c, line_color='g', ax=None, display=True)
