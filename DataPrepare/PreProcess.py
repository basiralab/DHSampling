import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import embed

# TODO: functions to convert RGB medmnist data to gray image (2D MedMNIST)
def _rgb_2_gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting for MedMNIST data preprocess')
    parser.add_argument('--medmnist_origin', type=str, default='/path/2/original/medmnist/path', help='specify original medmnist data file')
    parser.add_argument('--medmnist_target', type=str, default='/path/2/target/medmnist/path', help='specify training file list')

    args = parser.parse_args()

    data_flags_2d = ['organcmnist', 'organsmnist']

    for data_flag in tqdm(data_flags_2d):
        target_folder = os.path.join(args.medmnist_target, data_flag)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        data_npz_path = os.path.join(args.medmnist_origin, data_flag+'.npz')
        data_npz = np.load(data_npz_path)

        data_training_image = data_npz['train_images']
        data_training_label = data_npz['train_labels']

        if len(data_training_image.shape) == 4:
            data_training_image_gray = np.zeros((data_training_image.shape[0], data_training_image.shape[1], data_training_image.shape[2]))
            num_training =  data_training_image.shape[0]

            for idx in range(num_training):
                tmp_rgb = data_training_image[idx, :, :, :]
                tmp_gray = _rgb_2_gray(tmp_rgb)
                data_training_image_gray[idx, :, :] = tmp_gray

            np.save(os.path.join(target_folder, 'train_images.npy'), data_training_image_gray)
            np.save(os.path.join(target_folder, 'train_labels.npy'), data_training_label)

        elif len(data_training_image.shape)==3:
            np.save(os.path.join(target_folder, 'train_images.npy'), data_training_image)
            np.save(os.path.join(target_folder, 'train_labels.npy'), data_training_label)

        else:
            print(data_flag)

        data_val_image = data_npz['val_images']
        data_val_label = data_npz['val_labels']

        if len(data_val_image.shape) == 4:
            data_val_image_gray = np.zeros((data_val_image.shape[0], data_val_image.shape[1], data_val_image.shape[2]))
            num_val =  data_val_image.shape[0]

            for idx in range(num_val):
                tmp_rgb = data_val_image[idx, :, :, :]
                tmp_gray = _rgb_2_gray(tmp_rgb)
                data_val_image_gray[idx, :, :] = tmp_gray

            np.save(os.path.join(target_folder, 'val_images.npy'), data_val_image_gray)
            np.save(os.path.join(target_folder, 'val_labels.npy'), data_val_label)

        elif len(data_val_image.shape) == 3:
            np.save(os.path.join(target_folder, 'val_images.npy'), data_val_image)
            np.save(os.path.join(target_folder, 'val_labels.npy'), data_val_label)

        else:
            print(data_flag)


        data_test_image = data_npz['test_images']
        data_test_label = data_npz['test_labels']

        if len(data_test_image.shape) == 4:
            data_test_image_gray = np.zeros((data_test_image.shape[0], data_test_image.shape[1], data_test_image.shape[2]))
            num_test =  data_test_image.shape[0]

            for idx in range(num_test):
                tmp_rgb = data_test_image[idx, :, :, :]
                tmp_gray = _rgb_2_gray(tmp_rgb)
                data_test_image_gray[idx, :, :] = tmp_gray

            np.save(os.path.join(target_folder, 'test_images.npy'), data_test_image_gray)
            np.save(os.path.join(target_folder, 'test_labels.npy'), data_test_label)

        elif len(data_test_image.shape) == 3:
            np.save(os.path.join(target_folder, 'test_images.npy'), data_test_image)
            np.save(os.path.join(target_folder, 'test_labels.npy'), data_test_label)

        else:
            print(data_flag)
