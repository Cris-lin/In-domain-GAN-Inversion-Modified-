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


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  #parser.add_argument('--model_path', type=str, default='styleganinv_face_256.pkl',
  #                    help='Path to the pre-trained model.')
  parser.add_argument('--model_path', type=str, default='network-final_L.pkl',
                      help='Path to the pre-trained model.')
  parser.add_argument('--image_list', type=str, default='./new_examples/test.list',
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/opt/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size. (default: 1)')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=1,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 1)')
  parser.add_argument('-R', '--random_init', action='store_true',
                      help='Whether to use random initialization instead of '
                           'the output from encoder. (default: False)')
  parser.add_argument('-E', '--domain_regularizer', action='store_false',
                      help='Whether to use domain regularizer for '
                           'optimization. (default: True)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')

  # FLAGS = parser.parse_args()  # 然后所有的命令放入FLAGS
  # print(FLAGS)
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = '0'

  assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  output_dir = args.output_dir or f'results/opt_500_modi_1/{image_list_name}'
  os.makedirs(output_dir, exist_ok=True)
  #logger = setup_logger(output_dir, 'inversion.log', 'inversion1_logger')

  #logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': int(time.time())})
  with open(args.model_path, 'rb') as f:
    E, _, _, Gs, _ = pickle.load(f)
    #E, _, _, Gs = pickle.load(f)

  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]

  # Build graph.
  #logger.info(f'Building graph.')
  sess = tf.get_default_session()
  perceptual_model = PerceptualModel([image_size, image_size], False)

  # defining meta data
  weight_for_feature = 1
  weight_for_enc = 1e-1
  epochs = 1000
  pix_learning_rate = 1e-2
  feat_learning_rate = 1e-2
  batch_size = 25

  # defining shapes
  input_shape = E.input_shape
  #input_shape[0] = args.batch_size
  input_shape[0] = batch_size
  latent_shape = Gs.components.synthesis.input_shape
  #latent_shape[0] = args.batch_size
  latent_shape[0] = batch_size


  
  # building computational graph
  #进行求逆的真实图片
  real_image = tf.placeholder( name = 'real_image', shape = input_shape, dtype = tf.float32 )
  #当前对应的潜变量
  #latent_Z = tf.get_variable( name = 'latent_Y', shape = latent_shape, dtype = tf.float32, initializer =tf.random_normal_initializer(mean=0, stddev=1) )
  latent_X = tf.get_variable( name = 'latent_X', shape = latent_shape, dtype = tf.float32, initializer =tf.random_normal_initializer(mean=0, stddev=1) )
  #Dual_Y = tf.get_variable( name = 'Dual_Y', shape = [batch_size, latent_shape[1]*latent_shape[2]], dtype = tf.float32, initializer =tf.random_normal_initializer(mean=0, stddev=1) )

  #对潜变量进行初始化操作
  init_latent_X = tf.assign(latent_X, tf.reshape(E.get_output_for( real_image, phase = False), latent_shape))
  #init_latent_Z = tf.assign(latent_Z, tf.reshape(E.get_output_for( real_image, phase = False), latent_shape))
  
  #将真实图片转化为VGG16的接受形式
  real_image_255 = (tf.transpose(real_image, [0, 2, 3, 1]) + 1) / 2 * 255
  #真实图片的自编码器输出
  real_encode = tf.reshape(E.get_output_for( real_image, phase = False), latent_shape)
  #真实图片的VGG16输出
  real_feat = perceptual_model( real_image_255 )

  #根据当前潜变量生成的图片
  #image_rec_Z = Gs.components.synthesis.get_output_for(latent_Z, randomize_noise=False)
  image_rec = Gs.components.synthesis.get_output_for(latent_X, randomize_noise=False)
  #将重建图片转化为VGG16的接受形式
  image_rec_255 = (tf.transpose(image_rec, [0, 2, 3, 1]) + 1) / 2 * 255
  #重建图片的VGG16输出
  feat_rec = perceptual_model( image_rec_255 )
  encode_rec = tf.reshape(E.get_output_for( image_rec, phase = False), latent_shape)

  Loss_rec = tf.reduce_mean( tf.square( real_encode - encode_rec ), axis = [1,2])
  #真实图片和重建图片基于逐个像素的误差
  Loss_pix = tf.reduce_mean( tf.square( image_rec - real_image ), axis = [1,2,3]) + weight_for_enc*Loss_rec
  #真实图片和重建图片基于VGG16提取特征的误差
  Loss_feat = tf.reduce_mean( tf.square( feat_rec - real_feat ), axis = [1]) + weight_for_enc*Loss_rec
  #Loss_Dual = tf.reshape(tf.diag_part(tf.matmul( Dual_Y, tf.transpose(tf.reshape(latent_X - latent_Z, [batch_size,-1])))), [batch_size, -1])
  #损失函数为三者误差总和
  Loss = Loss_pix + weight_for_feature*Loss_feat# + weight_for_Aug*Loss_Aug# + Loss_Dual

  #更新潜变量
  optimizer_feat = tf.train.AdamOptimizer(feat_learning_rate)
  feat_train_op = optimizer_feat.minimize(Loss_feat, var_list=[latent_X])
  optimizer_pix = tf.train.AdamOptimizer(pix_learning_rate)
  pix_train_op = optimizer_pix.minimize(Loss_pix, var_list=[latent_X])
  #Y_train_op = tf.assign( Dual_Y, Dual_Y + weight_for_Aug*tf.reshape((latent_X - latent_Z), [batch_size, -1]))
  tflib.init_uninitialized_vars()

  #损失分数
  score = Loss_pix + weight_for_feature*Loss_feat

  # Load image list.
  #logger.info(f'Loading image list.')
  image_list = []
  with open(args.image_list, 'r') as f:
    for line in f:
      image_list.append(line.strip())

  # Invert images.
  #logger.info(f'Start inversion.')
  save_interval = epochs // args.num_results
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, epochs + 1):
    if step == epochs or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  #viz_size = None if args.viz_size == 0 else args.viz_size
  #visualizer = HtmlPageVisualizer(
  #    num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  #visualizer.set_headers(headers)

  images = np.zeros(input_shape, np.uint8)
  names = ['' for _ in range(batch_size)]
  latent_codes_enc = []
  latent_codes = []
  loss_list = []
  #for img_idx in tqdm(range(0, len(image_list), batch_size), leave=False):
  for img_idx in range(0, len(image_list), batch_size):
    # Load inputs.
    batch = image_list[img_idx:img_idx + batch_size]
    for i, image_path in enumerate(batch):
      image = resize_image(load_image(image_path), (image_size, image_size))
      images[i] = np.transpose(image, [2, 0, 1])
      names[i] = os.path.splitext(os.path.basename(image_path))[0]
    inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
    # Run encoder.
    sess.run([init_latent_X], {real_image:inputs})
    encode = latent_X.eval()
    encoder_image = sess.run(image_rec)
    latent_codes_enc.append(encode[0:len(batch)])
    encoder_image = adjust_pixel_range(encoder_image)
    for i, _ in enumerate(batch):
      image = np.transpose(images[i], [1, 2, 0])
      save_image(f'{output_dir}/{names[i]}_ori.png', image)
      save_image(f'{output_dir}/{names[i]}_enc.png', encoder_image[i])
    #  visualizer.set_cell(i + img_idx, 0, text=names[i])
    #  visualizer.set_cell(i + img_idx, 1, image=image)
    #  visualizer.set_cell(i + img_idx, 2, image=encoder_image[i])

    
  #  col_idx = 3
    #print('initialing')
    loss_list = []
  #  for step in tqdm(range(1, epochs+1), leave=False):
    for step in range(1, epochs+1):
      
      sess.run(pix_train_op, {real_image:inputs})
      sess.run(feat_train_op, {real_image:inputs})
      #sess.run(Y_train_op, {real_image:inputs})
      #cur_loss = sess.run(Loss, {real_image:inputs})

      #loss_list.append(cur_loss)

      if step >= epochs:
          '''
          plt.figure()
          plt.xlabel("iterations") 
          plt.ylabel("log-loss") 
          plt.semilogy(range(1,epochs+1), loss_list)
          plt.savefig(f'{output_dir}/{names[i]}_loss.png')
          '''
          latent_codes.append(latent_X.eval())
          outputs = sess.run(image_rec, feed_dict = {real_image:inputs})
          outputs = adjust_pixel_range(outputs)
          for i, _ in enumerate(batch):
  #          visualizer.set_cell(i + img_idx, col_idx, image=outputs[i])
            save_image(f'{output_dir}/{names[i]}_inv.png', outputs[i])
  #        col_idx += 1

  #logger.info(f'inversion finished and saving results.')
  # Save results.
  os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/encoded_codes.npy',
      np.concatenate(latent_codes_enc, axis=0))
  np.save(f'{output_dir}/inverted_codes.npy',
      np.concatenate(latent_codes, axis=0))
  #visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
  main()