import os

DATASET_PATH='/scratch/ll21/ys7429/pcpnet/'
LOGDIR = '/scratch/ll21/ys7429/log/jetnet_nci_new3/ablations/'
# DATASET_PATH='/home/sitzikbs/Datasets/pcpnet/'
# LOGDIR = './log/jetnet_nci_new2/experiments/'

BATCH_SIZE = 1024
TRAIN_SET = 'trainingset_whitenoise.txt'
TESTSET = 'testset_all.txt'
GPUIDX = 0 # must be 0 on server
N_EPOCHS = 600
N_GAUSSIANS = 1
N_POINTS = 384
ORDER = 3
LR=0.001
SCHEDULER="step"
arch = "simple"
CON_REG="log"
WEIGHT_MODE='sigmoid'
LOSS_TYPE='sin'
NN_SEARCH="k"
USE_CONSISTENCY=1
USE_POINT_STN=1
OPT_EPS=1e-8
COMPUTE_RESIDUALS=0
N_EXPERTS = 3
name = "Deepfit_" + arch + "_" + WEIGHT_MODE + "_cr_" + CON_REG + "_d" + str(ORDER) + "_p" + str(N_POINTS) + "_L" + LOSS_TYPE
# name = "Deepfit_" + arch + "_d" + str(ORDER) + "_p" + str(N_POINTS) + "_g" + str(N_GAUSSIANS)

print("training {}".format(name))
os.system('python3 train_n_est.py --indir {} --name {} --points_per_patch {} --gpu_idx {} --batchSize {} --jet_order {} '
          '--nepoch {} --fit_type {} --trainset {} --testset {} --logdir {} --n_gaussians {} --arch {} --normal_loss {} '
          '--weight_mode {} --use_consistency {} --saveinterval 20 --use_point_stn {} --lr {} --opt_eps {} --con_reg {}'
          ' --scheduler_type {} --neighbor_search {} --compute_residuals {} --n_experts {}'
            # ' --refine --refine_epoch 360'
          .format(DATASET_PATH, name, N_POINTS, GPUIDX, BATCH_SIZE, ORDER, N_EPOCHS, 'jet', TRAIN_SET,
                  'validationset_no_noise.txt', LOGDIR, N_GAUSSIANS, arch, LOSS_TYPE, WEIGHT_MODE, USE_CONSISTENCY,
                  USE_POINT_STN, LR, OPT_EPS, CON_REG, SCHEDULER, NN_SEARCH, COMPUTE_RESIDUALS, N_EXPERTS))


print("testing {}".format(name))
os.system('python3 test_n_est.py --testset {} --modelpostfix {} --logdir {} --gpu_idx {} --models {} --export_visualization 0'.format(
    TESTSET, "_model_" + str(N_EPOCHS-1) + ".pth", LOGDIR, GPUIDX, name))

print("evaluating {}".format(name))
# os.system('python3 evaluate.py --normal_results_path {} --dataset_list {}'.format("./log/jetnet/"+name+"/results/", 'testset_no_noise'))
os.system('python3 evaluate.py --normal_results_path {} --dataset_list {} {} {} {} {} {}'.format(LOGDIR+name+"/results/", 'testset_no_noise',  'testset_low_noise', 'testset_med_noise', 'testset_high_noise',
                             'testset_vardensity_striped', 'testset_vardensity_gradient'))

