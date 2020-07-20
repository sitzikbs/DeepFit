# run_DeepFit_single_experiment.py run a full train, test, evaluate for given parameters 
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

# train, test and evaluate DeepFit
import os

DATASET_PATH='./data/pcpnet/'
LOGDIR = './log/my_experiments/'

BATCH_SIZE = 512
TRAIN_SET = 'trainingset_whitenoise.txt'
VAL_SET = 'validationset_no_noise.txt'
TESTSET = 'testset_all.txt'
GPUIDX = 0  # must be 0 on server
N_EPOCHS = 600
N_GAUSSIANS = 1
N_POINTS = 256
ORDER = 3
LR=0.001
SCHEDULER="step"
arch = "simple"
CON_REG="log"
WEIGHT_MODE='sigmoid'
LOSS_TYPE='sin'
NN_SEARCH="k"

COMPUTE_RESIDUALS=0

name = "my_experiment"

print("training {}".format(name))
os.system('python3 train_n_est.py --indir {} --name {} --points_per_patch {} --gpu_idx {} --batchSize {} --jet_order {} '
          '--nepoch {} --trainset {} --testset {} --logdir {} --n_gaussians {} --arch {} --normal_loss {} '
          '--weight_mode {} --saveinterval 20 --lr {} --con_reg {}'
          ' --scheduler_type {} --neighbor_search {}'
          .format(DATASET_PATH, name, N_POINTS, GPUIDX, BATCH_SIZE, ORDER, N_EPOCHS, TRAIN_SET,
                  VAL_SET, LOGDIR, N_GAUSSIANS, arch, LOSS_TYPE, WEIGHT_MODE,
                  LR, CON_REG, SCHEDULER, NN_SEARCH))

print("testing {}".format(name))
os.system('python3 test_n_est.py --testset {} --modelpostfix {} --logdir {} --gpu_idx {} --models {}'.format(
    TESTSET, "_model_" + str(N_EPOCHS-1) + ".pth", LOGDIR, GPUIDX, name))

print("evaluating {}".format(name))
os.system('python3 evaluate.py --normal_results_path {} --dataset_list {} {} {} {} {} {}'.format(LOGDIR+name+"/results/", 'testset_no_noise',  'testset_low_noise', 'testset_med_noise', 'testset_high_noise',
                             'testset_vardensity_striped', 'testset_vardensity_gradient'))
