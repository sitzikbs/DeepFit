# cgal_normal_estimation.py cgal normal estimation baselines
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

from __future__ import print_function
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_processing_3 import *
import os
import numpy as np

n_neighbors_list = [128, 256, 512, 1024] #[18, 112, 450] #18, 112, 450
methods = ['pca', 'jet']
sparse_patches = False

BASELINE_DIR = os.path.dirname('/mnt/sitzikbs_storage/PycharmProjects/DeepFit3D/Normal_Estimation/log/baselines/')
BASE_DIR = os.path.abspath(os.path.join(BASELINE_DIR, os.pardir))
data_dir = os.path.join(BASE_DIR, '/home/sitzikbs/Datasets/NYU_V2/')
file_list_file = os.path.join(data_dir, 'testset_all.txt')
with open(file_list_file) as f:
       file_list = f.readlines()
file_list = [x.strip() for x in file_list]
file_list = list(filter(None, file_list))
for n_neighbors in n_neighbors_list:
    for method in methods:
        output_path = os.path.join(BASELINE_DIR, os.path.join(method, str(n_neighbors) + '_neighbors'))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for filename in file_list:
            normals_filename = os.path.join(output_path, filename + '.normals')
            if not os.path.exists(normals_filename):
                points = []
                normals = []
                print("Reading file " + filename + "...")
                read_xyz_points(os.path.join(data_dir, filename + '.xyz'), points)
                print(len(points), " points read")
                if sparse_patches:
                    points_idx = np.loadtxt(os.path.join(data_dir, filename + '.pidx')).astype('int')
                    new_points = []
                    for i in points_idx:
                        new_points.append(points[i])
                    points = new_points

                if method == 'jet':
                    print("Running jet_estimate_normals...")
                    jet_estimate_normals(points, normals, n_neighbors)
                    print(len(normals), " normal vectors set")
                elif method == 'pca':
                    print("Running pca_estimate_normals...")
                    pca_estimate_normals(points, normals, n_neighbors)
                    print(len(normals), " normal vectors set")
                else:
                    print('Method not supported')
                py_normals = []
                for n in normals:
                    py_normals.append([n.x(), n.y(), n.z()])
                py_normals = np.array(py_normals)
                np.savetxt(normals_filename, py_normals)
