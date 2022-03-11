'''
Created at 20210511
@author:CrisLin
'''

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib
import time
import matplotlib.pyplot as plt

from perceptual_model import PerceptualModel
from utils.logger import setup_logger
from utils.visualizer import adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image



def main():

    ##global setting
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    tflib.init_tf({'rnd.np_random_seed': int(time.time())})

    ##environ setting
    image_list_dir = f'new_examples/test.list'
    model_path = f'styleganinv_face_256.pkl'
    image_list_name = os.path.splitext(os.path.basename(image_list_dir))[0]
    output_dir = f'results/sampling_500'
    os.makedirs(output_dir, exist_ok=True)

    ##loading model
    with open(model_path, 'rb') as f:
        E, _, _, Gs = pickle.load(f)

    ##meta data
    batch_size = 1
    epochs = 500
    sampling_size = 100
    weight_for_feature = 5e-2
    variance = 1

    ##defining shapes
    image_shape = E.input_shape
    image_shape[0] = batch_size
    input_shape = Gs.components.mapping.input_shape
    input_shape[0] = batch_size
    latent_shape = Gs.components.synthesis.input_shape
    latent_shape[0] = batch_size
    image_size = E.input_shape[2]

    ##initilization
    sess = tf.get_default_session()
    perceptual_model = PerceptualModel([image_size, image_size], False)

    ##defining computing graph

    #original latent code from Gaussian Distribution
    Gaussian_latent_code = tf.get_variable(name='Glatent_code', shape=input_shape, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
    #real image to reconstrution
    real_image = tf.placeholder(name='real_image', shape=image_shape, dtype=tf.float32)
    #variance for sampling
    variance = tf.placeholder(name='variance', shape=(), dtype=tf.float32)
    #random disturb of MALA
    random_eps = tf.placeholder(name='eps', shape=input_shape, dtype=tf.float32)
    #update creteria of Metropolis-Hastings
    update_creteria = tf.placeholder(name='update_creteria', shape=(), dtype=tf.float32)
    #step size
    step_size = tf.placeholder(name='update_creteria', shape=(), dtype=tf.float32)
    #latent code in latent space
    latent_code = Gs.components.mapping.get_output_for(Gaussian_latent_code, None)
    #initialization of latent code
    init_latent_code = tf.assign(Gaussian_latent_code, tf.random_normal(mean=0.0, stddev=1.0, shape=input_shape))

    #convert real image to input style for VGG
    real_image_255 = (tf.transpose(real_image, [0, 2, 3, 1]) + 1) / 2 * 255
    #feature output of real image with VGG
    real_feat = perceptual_model( real_image_255 )

    #根据当前潜变量生成的图片
    image_rec = Gs.components.synthesis.get_output_for(latent_code, randomize_noise=False)
    #将重建图片转化为VGG16的接受形式
    image_rec_255 = (tf.transpose(image_rec, [0, 2, 3, 1]) + 1) / 2 * 255
    #重建图片的VGG16输出
    feat_rec = perceptual_model( image_rec_255 )

    #真实图片和重建图片基于逐个像素的误差
    Loss_pix = tf.reduce_mean( tf.square( image_rec - real_image ))#, axis = [1,2,3])
    #真实图片和重建图片基于VGG16提取特征的误差
    Loss_feat = tf.reduce_mean( tf.square( feat_rec - real_feat ))#, axis = [1])
    #损失函数为三者误差总和
    Loss = Loss_pix + weight_for_feature*Loss_feat

    #损失函数对潜变量的梯度
    latent_code_gradient = tf.reshape(tf.gradients(Loss, Gaussian_latent_code ), input_shape)
    #latent_code_gradient = tf.gradients(Loss, latent_code)[0]
    #根据MALA算法得出的新样本
    sample_latent_code = Gaussian_latent_code - tf.multiply(step_size, latent_code_gradient) + random_eps

    #新样本潜变量生成的图片
    image_sample = Gs.components.synthesis.get_output_for(Gs.components.mapping.get_output_for(sample_latent_code, None), randomize_noise=False)
    #将新样本图片转化为VGG16的接受形式
    image_sample_255 = (tf.transpose(image_sample, [0, 2, 3, 1]) + 1) / 2 * 255
    #新样本图片的VGG16输出
    feat_sample = perceptual_model( image_sample_255 )

    #新样本图片的像素误差
    Loss_pix_sample = tf.reduce_mean( tf.square( image_sample - real_image ))#, axis = [1,2,3])
    #新样本图片的特征误差
    Loss_feat_sample = tf.reduce_mean( tf.square( feat_sample - real_feat ))#, axis = [1])
    #新样本的误差总和
    Loss_sample = Loss_pix_sample + weight_for_feature*Loss_feat_sample

    #Metropolis-Hastings接受拒绝采样
    Probabilty = tf.exp(-Loss/variance)
    Probabilty_sample = tf.exp(-Loss_sample/variance)

    new_latent_code = tf.cond( tf.less((Probabilty/Probabilty_sample), update_creteria), lambda:Gaussian_latent_code, lambda:sample_latent_code)

    #更新潜变量
    train_op = tf.assign(Gaussian_latent_code, new_latent_code)

    tflib.init_uninitialized_vars()



    ##importing images
    image_list = []
    with open(image_list_dir, 'r') as f:
        for line in f:
            image_list.append(line.strip())

    images = np.zeros(image_shape, np.uint8)
    names = ['' for _ in range(batch_size)]
    latent_codes = []
    for img_idx in tqdm(range(0, len(image_list), batch_size), leave=False):
    #for img_idx in tqdm(range(0, 1, batch_size), leave=False):
        batch = image_list[img_idx:img_idx + batch_size]
        for i, image_path in enumerate(batch):
            image = resize_image(load_image(image_path), (image_size, image_size))
            images[i] = np.transpose(image, [2, 0, 1])
            names[i] = os.path.splitext(os.path.basename(image_path))[0]

        inputs = images.astype(np.float32) / 255 * 2.0 - 1.0

        for i, _ in enumerate(batch):
            image = np.transpose(images[i], [1, 2, 0])
            save_image(f'{output_dir}/{names[i]}_ori.png', image)

    ##start sampling
        sess.run(init_latent_code, {real_image:inputs})
        cur_step_size = 1e-2
        cur_var = 1
        for step in tqdm(range(1, epochs+1), leave=False):
            if step % 10 == 0:
                cur_step_size = cur_step_size*0.95
            cur_eps = np.random.normal(0, np.sqrt(2*cur_step_size), input_shape)
            cur_uc = np.random.uniform(0, 1)
            sess.run(train_op, feed_dict = {real_image:inputs, variance:cur_var, random_eps:cur_eps, update_creteria:cur_uc, step_size:cur_step_size})

        Sample_list = []
        for sample_idx in tqdm(range(1, sampling_size+1), leave=False):
            cur_eps = np.random.normal(0, np.sqrt(2*cur_step_size), input_shape)
            cur_uc = np.random.uniform(0, 1)
            sess.run(train_op, feed_dict = {real_image:inputs, variance:cur_var, random_eps:cur_eps, update_creteria:cur_uc, step_size:cur_step_size})
            Sample_list.append(Gaussian_latent_code.eval())


        avg_latent_code = np.mean(Sample_list, axis=0)
        latent_codes.append(avg_latent_code)
        #outputs = sess.run(image_rec, feed_dict = {real_image:inputs, Gaussian_latent_code:avg_latent_code})
        #outputs = adjust_pixel_range(outputs)
        #for i, _ in enumerate(batch):
        #    save_image(f'{output_dir}/{names[i]}_sample.png', outputs[i])

        
    os.system(f'cp {image_list_dir} {output_dir}/image_list.txt')
    np.save(f'{output_dir}/inverted_codes.npy',
        np.concatenate(latent_codes, axis=0))


if __name__ == '__main__':
    main()
