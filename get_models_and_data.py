import os
import zipfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_and_unzip(source_url, target_file, target_dir):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    global downloaded
    downloaded = 0
    def show_progress(count, block_size, total_size):
        global downloaded
        downloaded = downloaded + block_size
        print('downloading ... %d%%\r' % round(((downloaded*100.0) / total_size)))

    print('downloading ... \r',)
    urllib.request.urlretrieve(source_url, filename=target_file, reporthook=show_progress)
    print('downloading ... done')

    print('unzipping ... \r')
    zip_ref = zipfile.ZipFile(target_file, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()
    os.remove(target_file)
    print('unzipping ... done')

def get_point_clouds(get_original_pcpnet=False):
    '''
    Download point clouds from PCPNet, NYU V2, ScanNet datasets. Datasets are sved to the `data/` dir
    :param get_original_pcpnet: True/False indicating if to get the pcpnet dataset from the original project or from ours,
    The main difference is that ours includes several noise level .txt files which enable training the switching network
    '''
    print('Fetching the data... \n')
    if get_original_pcpnet:
        # get pcpnet data from original project dir
        source_url = 'http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip'
        target_dir = os.path.join(BASE_DIR, 'data/pcpnet/')
        target_file = os.path.join(target_dir, 'pclouds.zip')
        download_and_unzip(source_url, target_file, target_dir)

        # get subset of nyu v2 and scannet for testing
        source_url = 'https://gitlab.cs.technion.ac.il/mic/nesti-net-resources/raw/master/data/nyu_scannet_pclouds.zip'
        target_dir = os.path.join(BASE_DIR, 'data/')
        target_file = os.path.join(target_dir, 'nyu_scannet_pclouds.zip')
        download_and_unzip(source_url, target_file, target_dir)

    else:
        # get a copy of the pcpnet dataset with noise level indicators for selected subsets
        source_url = 'https://gitlab.cs.technion.ac.il/mic/nesti-net-resources/raw/master/data.zip'
        target_dir = os.path.join(BASE_DIR, 'data/')
        target_file = os.path.join(target_dir, 'data.zip')
        download_and_unzip(source_url, target_file, target_dir)


def get_trained_model(fetch_all=False):

    print('Fetching the trained models... \n')
    if fetch_all:
        source_url = 'https://gitlab.cs.technion.ac.il/mic/nesti-net-resources/raw/master/log/all_model_logs.zip'
        target_dir = os.path.join(BASE_DIR, 'log/')
        target_file = os.path.join(target_dir, 'all_model_logs.zip')
    else:
        source_url = 'https://gitlab.cs.technion.ac.il/mic/nesti-net-resources/raw/master/log/experts.zip'
        target_dir = os.path.join(BASE_DIR, 'log/')
        target_file = os.path.join(target_dir, 'experts.zip')

    download_and_unzip(source_url, target_file, target_dir)


def get_point_clouds_and_models_from_repo(repo_url):
    os.system('git clone %s' % (repo_url))


if __name__ == '__main__':

    get_point_clouds(get_original_pcpnet=False)
    get_trained_model(fetch_all=False)