import os
import shutil


data_base_dir = '.'

if not os.path.isdir(os.path.join(data_base_dir, 'data/full_data/train')):
    if not os.path.exists(os.path.join(data_base_dir, 'data/dataset_full.zip')):
        raise Warning('Please download dataset_full.zip first!')
    else:
        shutil.unpack_archive(os.path.join(data_base_dir, 'data/dataset_full.zip'),
                              extract_dir=os.path.join(data_base_dir, 'data'), format='zip')
        os.makedirs(os.path.join(data_base_dir, 'data/full_data/train'), exist_ok=True)
        os.makedirs(os.path.join(data_base_dir, 'data/full_data/test'), exist_ok=True)
        shutil.move(os.path.join(data_base_dir, 'data/work/train_sorted.csv'),
                    os.path.join(data_base_dir, 'data/full_data/train/'))
        shutil.move(os.path.join(data_base_dir, 'data/work/test.csv'),
                    os.path.join(data_base_dir, 'data/full_data/test/'))
        shutil.rmtree(os.path.join(data_base_dir, 'data/work'))
else:
    print('data/full_data/train already exist.')
