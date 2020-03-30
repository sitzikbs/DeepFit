import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

testset_list_file_path = '/home/itzik/Datasets/nyu v2/nyu_v2_txt/testset_file_list.txt'

test_set_files = []
with open(testset_list_file_path) as f:
    test_set_files = f.readlines()
test_set_files = [x.strip() for x in test_set_files]
test_set_files = list(filter(None, test_set_files))
print(test_set_files)


for testset in test_set_files:
    testset = os.path.basename(testset)
    command_string = 'python ' + BASE_DIR + '/test_n_est_w_experts.py --results_path="log/my_experts_kinect/" ' \
                     '--model="experts_n_est" --dataset_name="" ' \
                     '--dataset_path="/home/itzik/Datasets/nyu v2/nyu_v2_txt/" ' \
                     '--testset=' + testset + ' --sparse_patches=0 --batch_size=128'
    os.chdir(BASE_DIR)
    os.system(command_string)  # bkp of train procedure
