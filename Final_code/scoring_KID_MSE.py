## calculating FID

import os
import pickle
import numpy as np
import scipy
import time
import tensorflow as tf
from dnnlib import tflib
from utils.visualizer import save_image, load_image, resize_image

def main():

    tflib.init_tf({'rnd.np_random_seed': int(time.time())})
    sess = tf.get_default_session()

    with open('network-final_L.pkl', 'rb') as f:
        E, _, _, Gs, _ = pickle.load(f)
        #E, _, _, Gs = pickle.load(f)
    
    #with open(f'styleganinv_face_256.pkl', 'rb') as f:
    #    E, _, _, _ = pickle.load(f)

    real_image_list = f'to_score/real.list'
    inv_image_list = f'to_score/inv.list'
    enc_image_list = f'to_score/enc.list'
    image_size = E.input_shape[2]


    with open(f'inception_v3_features.pkl', 'rb') as f:
        inception = pickle.load(f)

    image_list = []
    with open(real_image_list, 'r') as f:
        for line in f:
            image_list.append(line.strip())

    images = []
    names = []
    for image_path in image_list:
        image = resize_image(load_image(image_path), (image_size, image_size))
        images.append(np.transpose(image, [2, 0, 1]))
        names.append(os.path.splitext(os.path.basename(image_path))[0])
    images = np.asarray(images, dtype=np.float32)
    #inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
    real_img = images.astype(np.float32) / 255 * 2.0 - 1.0

    real_activations = inception.run(images, assume_frozen=True)
    #real_activations = inception.run(real_img, assume_frozen=True)
    mu_real = np.mean(real_activations, axis = 0)
    #sigma_real = np.cov(real_activations, rowvar=False)

    image_list = []
    with open(inv_image_list, 'r') as f:
        for line in f:
            image_list.append(line.strip())

    images = []
    names = []
    for image_path in image_list:
        image = resize_image(load_image(image_path), (image_size, image_size))
        images.append(np.transpose(image, [2, 0, 1]))
        names.append(os.path.splitext(os.path.basename(image_path))[0])
    images = np.asarray(images, dtype=np.float32)
    #inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
    inv_img = images.astype(np.float32) / 255 * 2.0 - 1.0

    inv_activations = inception.run(images, assume_frozen=True)
    #inv_activations = inception.run(inv_img, assume_frozen=True)
    mu_inv = np.mean(inv_activations, axis = 0)
    #sigma_inv = np.cov(inv_activations, rowvar=False)

    #m = np.square(mu_inv - mu_real).sum()
    #s, _ = scipy.linalg.sqrtm(np.dot(sigma_inv, sigma_real), disp=False) # pylint: disable=no-member
    #dist = m + np.trace(sigma_inv + sigma_real - 2*s)

    m = np.matmul(np.transpose(mu_inv), mu_real)
    d = (mu_inv.shape)[0]
    print('重建KID为：', np.power(m/d+1, 3), 'MSE为', np.mean(np.square(inv_img - real_img)))

    image_list = []
    with open(enc_image_list, 'r') as f:
        for line in f:
            image_list.append(line.strip())

    images = []
    names = []
    for image_path in image_list:
        image = resize_image(load_image(image_path), (image_size, image_size))
        images.append(np.transpose(image, [2, 0, 1]))
        names.append(os.path.splitext(os.path.basename(image_path))[0])
    images = np.asarray(images, dtype=np.float32)
    #inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
    enc_img = images.astype(np.float32) / 255 * 2.0 - 1.0

    enc_activations = inception.run(images, assume_frozen=True)
    #enc_activations = inception.run(enc_img, assume_frozen=True)
    mu_enc = np.mean(enc_activations, axis = 0)
    #sigma_enc = np.cov(enc_activations, rowvar=False)

    #m = np.square(mu_enc - mu_real).sum()
    #s, _ = scipy.linalg.sqrtm(np.dot(sigma_enc, sigma_real), disp=False) # pylint: disable=no-member
    #dist = m + np.trace(sigma_enc + sigma_real - 2*s)

    m = np.matmul(np.transpose(mu_enc), mu_real)
    d = (mu_enc.shape)[0]

    print('encoder KID为：', np.power(m/d+1, 3), 'MSE为', np.mean(np.square(enc_img - real_img)))


if __name__ == '__main__':
  main()