# Download pcpnet data
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

def get_pcpnet_point_clouds():
    '''
    Download point clouds from PCPNet datasets. files are saved to the `data/` dir
    '''
    print('Fetching the data... \n')

    # get pcpnet data from original project dir
    source_url = 'http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip'
    target_dir = os.path.join(BASE_DIR, 'data/pcpnet/')
    target_file = os.path.join(target_dir, 'pclouds.zip')
    download_and_unzip(source_url, target_file, target_dir)

def get_point_clouds_and_models_from_repo(repo_url):
    os.system('git clone %s' % (repo_url))

if __name__ == '__main__':
    get_pcpnet_point_clouds()
