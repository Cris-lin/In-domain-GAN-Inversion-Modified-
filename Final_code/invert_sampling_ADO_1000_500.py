# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

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
    image_dir = f'results/sampling_500'
    image_list_path = f'{image_dir}/image_list.txt'
    latent_path = f'{image_dir}/inverted_codes.npy'
    model_path = f'styleganinv_face_256.pkl'
    image_list_name = os.path.splitext(os.path.basename(image_list_path))[0]
    output_dir = f'results/inversion_sampling_500'
    os.makedirs(output_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
      	E, _, _, Gs = pickle.load(f)

    #meta data
    weight_for_feature = 2
    epochs = 1000
    pix_learning_rate = 1e-2
    feat_learning_rate = 1e-2
    batch_size = 25

    # defining shapes
    image_shape = E.input_shape
    image_shape[0] = batch_size
    input_shape = Gs.components.mapping.input_shape
    input_shape[0] = batch_size
    latent_shape = Gs.components.synthesis.input_shape
    latent_shape[0] = batch_size
    image_size = E.input_shape[2]

    # Build graph.
    sess = tf.get_default_session()
    perceptual_model = PerceptualModel([image_size, image_size], False)
    init_latent_codes = np.load(latent_path)

  
    # building computational graph
    #进行求逆的真实图片
    real_image = tf.placeholder( name = 'real_image', shape = image_shape, dtype = tf.float32 )
    init_latent_code = tf.placeholder( name = 'real_image', shape = input_shape, dtype = tf.float32 )
    #当前对应的潜变量
    latent_X = tf.get_variable( name='latent_X', shape=latent_shape, dtype = tf.float32, initializer =tf.random_normal_initializer(mean=0, stddev=1) )
  
    #将真实图片转化为VGG16的接受形式
    real_image_255 = (tf.transpose(real_image, [0, 2, 3, 1]) + 1) / 2 * 255
    #真实图片的VGG16输出
    real_feat = perceptual_model( real_image_255 )

    #对潜变量进行初始化操作
    init_latent_X = tf.assign(latent_X, Gs.components.mapping.get_output_for(init_latent_code, None))

    #根据当前潜变量生成的图片
    image_rec = Gs.components.synthesis.get_output_for(latent_X, randomize_noise=False)
    #将重建图片转化为VGG16的接受形式
    image_rec_255 = (tf.transpose(image_rec, [0, 2, 3, 1]) + 1) / 2 * 255
    #重建图片的VGG16输出
    feat_rec = perceptual_model( image_rec_255 )

    #真实图片和重建图片基于逐个像素的误差
    Loss_pix = tf.reduce_mean( tf.square( image_rec - real_image ), axis = [1,2,3])
    #真实图片和重建图片基于VGG16提取特征的误差
    Loss_feat = tf.reduce_mean( tf.square( feat_rec - real_feat ), axis = [1])
    #损失函数为三者误差总和
    Loss = Loss_pix + weight_for_feature*Loss_feat

    #更新潜变量
    optimizer_feat = tf.train.AdamOptimizer(feat_learning_rate)
    feat_train_op = optimizer_feat.minimize(Loss_feat, var_list=[latent_X])
    optimizer_pix = tf.train.AdamOptimizer(pix_learning_rate)
    pix_train_op = optimizer_pix.minimize(Loss_pix, var_list=[latent_X])
    tflib.init_uninitialized_vars()

    #损失分数
    score = Loss_pix + weight_for_feature*Loss_feat

    # Load image list.
    image_list = []
    with open(image_list_path, 'r') as f:
    	for line in f:
            image_list.append(line.strip())

  	# Invert images.
    save_interval = epochs
    headers = ['Name', 'Original Image', 'Warming Up Sample']
    for step in range(1, epochs + 1):
        if step == epochs or step % save_interval == 0:
            headers.append(f'Step {step:06d}')
    #    viz_size = 256
    #    visualizer = HtmlPageVisualizer(
    #  		num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
    #    visualizer.set_headers(headers)

    images = np.zeros(image_shape, np.uint8)
    names = ['' for _ in range(batch_size)]
    latent_codes = []
    for img_idx in tqdm(range(0, len(image_list), batch_size), leave=False):
    #for img_idx in range(0, len(image_list), batch_size):
    	# Load inputs.
        batch = image_list[img_idx:img_idx + batch_size]
        latent_batch = init_latent_codes[img_idx:img_idx + batch_size]
        for i, image_path in enumerate(batch):
            image = resize_image(load_image(image_path), (image_size, image_size))
            images[i] = np.transpose(image, [2, 0, 1])
            names[i] = os.path.splitext(os.path.basename(image_path))[0]
        inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
    	# Run encoder.
        sess.run([init_latent_X], {init_latent_code:latent_batch})
        sample_image = sess.run(image_rec)
        sample_image = adjust_pixel_range(sample_image)
        for i, _ in enumerate(batch):
            image = np.transpose(images[i], [1, 2, 0])
            save_image(f'{output_dir}/{names[i]}_ori.png', image)
            save_image(f'{output_dir}/{names[i]}_sample.png', sample_image[i])
    #        visualizer.set_cell(i + img_idx, 0, text=names[i])
    #        visualizer.set_cell(i + img_idx, 1, image=image)
    #        visualizer.set_cell(i + img_idx, 2, image=sample_image[i])

    
    #    col_idx = 3
        for step in tqdm(range(1, epochs+1), leave=False):
    #    for step in range(1, epochs+1):
            sess.run(pix_train_op, {real_image:inputs})
            sess.run(feat_train_op, {real_image:inputs})

            if step >= epochs:
                latent_codes.append(latent_X.eval())
                outputs = sess.run(image_rec, feed_dict = {real_image:inputs})
                outputs = adjust_pixel_range(outputs)
                for i, _ in enumerate(batch):
    #            visualizer.set_cell(i + img_idx, col_idx, image=outputs[i])
                    save_image(f'{output_dir}/{names[i]}_inv.png', outputs[i])
    #        col_idx += 1

  	# Save results.
    os.system(f'cp {image_list_path} {output_dir}/image_list.txt')
    np.save(f'{output_dir}/inverted_codes.npy',
        np.concatenate(latent_codes, axis=0))
    #visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
    main()