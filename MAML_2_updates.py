"""Data loading scripts"""
import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import scipy.io
import gc
from datetime import datetime

from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

COMPUTE_DTYPE = tf.float32
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

# COMPUTE_DTYPE = tf.float16


def get_mat_paths(paths, n_samples=None, shuffle=True):
  """
  Takes a set of data folders and labels and returns paths to image files
  paired with labels.
  Args:
    paths: A list of data folders
    labels: List or numpy array of same length as paths
    n_samples: Number of images to retrieve per data
  Returns:
    List of (label, image_path) tuples
  """
  if n_samples is not None:
    sampler = lambda x: random.sample(x, n_samples)
  else:
    sampler = lambda x: x
  mat_paths = [os.path.join(path, image)
           for path in paths
           for image in sampler(os.listdir(path))]
  if shuffle:
    random.seed(datetime.now())
    random.shuffle(mat_paths)
  return mat_paths


def mat_to_input_and_field(filename):
  """
  Takes an mat path and returns numpy array of input (1, 200) and output field (50, 200)
  Args:
    filename: Image filename
    dim_input: Flattened shape of image
  Returns:
    1 channel image
  """
  mat = scipy.io.loadmat(filename)
  Hr = np.array(mat['Hy_real'], dtype=np.float32)
  Hi = np.array(mat['Hy_imag'], dtype=np.float32)
  fields = np.stack((Hr, Hi), axis=2)

  img = np.array(mat['img'], dtype=np.float16)
  
  # TO DO: change 50 to a variable
  elongated_img = np.expand_dims(np.tile(img, (50,1)), axis=-1) # 1 channel
     
  # image = image.reshape([dim_input])
  # image = image.astype(np.float32) / 255.0
  # image = 1.0 - image
  return elongated_img, fields


class DataGenerator(object):
  """
  Data Generator capable of generating batches of EM_simulation data.
  A "class" is considered a class of simulation setup(i.e. x wavelength, x bars).
  """

  def __init__(self, num_class, num_samples_per_class, num_meta_test_class, num_meta_test_samples_per_class, config={}):
    """
    Args:
      num_classes: Number of classes within one frequency  (for now equals 1: regard all as same class within one fre)
      num_samples_per_class: num samples to generate per class in one batch (K-way)
      num_meta_test_classes: Number of classes within one frequency at meta-test time
      num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
      batch_size: size of meta batch size (e.g. number of functions)
    """
    self.num_classes = num_class
    self.num_samples_per_class = num_samples_per_class
    self.num_meta_test_classes = num_meta_test_class
    self.num_meta_test_samples_per_class = num_meta_test_samples_per_class

    data_folder = config.get('data_folder', '/scratch/users/chenkaim/data/')
    self.img_size = config.get('img_size', (50, 200))

    data_folders = [os.path.join(data_folder, wavelength, bars)
               for wavelength in os.listdir(data_folder)
               if os.path.isdir(os.path.join(data_folder, wavelength))
               for bars in os.listdir(os.path.join(data_folder, wavelength))
               if os.path.isdir(os.path.join(data_folder, wavelength, bars))]

    random.seed(321)
    random.shuffle(data_folders)
    num_val = int(0.2*len(data_folders))
    num_train = int(0.7*len(data_folders))
    self.metatrain_data_folders = data_folders[: num_train]
    self.metaval_data_folders = data_folders[
      num_train:num_train + num_val]
    self.metatest_data_folders = data_folders[
      num_train + num_val:]

  def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
    """
    Samples a batch for training, validation, or testing
    Args:
      batch_type: meta_train/meta_val/meta_test
      shuffle: randomly shuffle classes or not
      swap: swap number of classes (N) and number of samples per class (K) or not
    Returns:
      A a tuple of (1) Image batch and (2) Label batch where
      image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap is False
      where B is batch size, K is number of samples per class, N is number of classes
    """
    if batch_type == "meta_train":
      folders = self.metatrain_data_folders
      num_classes = self.num_classes
      num_samples_per_class = self.num_samples_per_class
    elif batch_type == "meta_val":
      folders = self.metaval_data_folders
      num_classes = self.num_classes
      num_samples_per_class = self.num_samples_per_class
    else:
      folders = self.metatest_data_folders
      num_classes = self.num_meta_test_classes
      num_samples_per_class = self.num_meta_test_samples_per_class
    all_image_batches, all_label_batches = [], []
    for i in range(batch_size):
      sampled_data_folders = random.sample(
        folders, num_classes)
      mat_paths = get_mat_paths(sampled_data_folders, n_samples=num_samples_per_class, shuffle=False)
      images_and_labels = [mat_to_input_and_field(
        li) for li in mat_paths]
      images = np.array([i[0] for i in images_and_labels])
      labels = np.array([i[1] for i in images_and_labels])
      # labels = np.array(labels).astype(np.int32)
      labels = np.reshape(labels, (num_classes, num_samples_per_class, self.img_size[0], self.img_size[1], NUM_OUT_CHANNELS))
      images = np.reshape(images, (num_classes, num_samples_per_class, self.img_size[0], self.img_size[1],1))
      #TO DO: rewrite shuffle using zip
      #batch = np.concatenate([labels, images], 2)
      #if shuffle:
      #  for p in range(num_samples_per_class):
      #    np.random.shuffle(batch[:, p])

      #labels = batch[:, :, :num_classes]
      #images = batch[:, :, num_classes:]

      if swap:
        labels = np.swapaxes(labels, 0, 1)
        images = np.swapaxes(images, 0, 1)

      all_image_batches.append(images)
      all_label_batches.append(labels)
    all_image_batches = np.stack(all_image_batches)
    all_label_batches = np.stack(all_label_batches)
    return all_image_batches, all_label_batches

"""Muskens architecture"""
import pandas as pd
import keras
#from keras.models import Model
from tensorflow import keras
from tensorflow.keras.backend import int_shape
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, ZeroPadding2D, Cropping2D, Concatenate, UpSampling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K

#from initializations import he_normal

from keras.utils.generic_utils import get_custom_objects
# DROP_RATE = 0.2
# NUM_EPOCHS = 200
ALPHA = 0.2
# LEARNING_RATE = 0.00005
# LR_DIV_FACTOR = 25.
# PCT_START = 0.2
# OMEGA = 1
# BATCH_SIZE = 16
NUM_DOWNCOV_BLOCKS = 5
NUM_UPSAMPLING_BLOCKS = NUM_DOWNCOV_BLOCKS-1
NUM_OUT_CHANNELS = 2
HIDDEN_DIM = 30
UPSAMPLE_INTERP = ["nearest", "bilinear"][1]
FNAME = '/scratch/users/chenkaim/test'

seed=4321
#FNAME = 's100_100_45k/leaky_alph02_b64mae_dr0_dc4_filsx1_lkw2_v7'

def conv_block(inp, cweight, bn, activation=LeakyReLU(alpha=ALPHA)):
  """ Perform, conv, batch norm, nonlinearity, and max pool """
  stride, no_stride = [1,2,2,1], [1,1,1,1]
  conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME')
  normed = bn(conv_output)
  output = activation(normed)
  return conv_output, normed, output

def res_Block(inp, cweights, bns, activation=LeakyReLU(alpha=ALPHA)):
    first_conv, _, x = conv_block(inp, cweights[0], bns[0], activation=LeakyReLU(alpha=ALPHA))
    _, _, x = conv_block(x, cweights[1], bns[1], activation=LeakyReLU(alpha=ALPHA))
    _, x, _ = conv_block(x, cweights[2], bns[2], activation=LeakyReLU(alpha=ALPHA))
    x = first_conv+x

    output = activation(x)

    return output

def _encodeBlock(inp, cweights, bns, activation=LeakyReLU(alpha=ALPHA)):
    res_out = res_Block(inp, cweights, bns, activation=LeakyReLU(alpha=ALPHA))
    pool_out = MaxPooling2D((2, 2), strides=2)(res_out)
    return res_out, pool_out

_upsample_matrix: tf.Tensor = tf.ones([2, 2, 1, 1],dtype=COMPUTE_DTYPE)
def upsample_helper(x: tf.Tensor) -> tf.Tensor:
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    batch, channels, height, width = x.shape
    x = tf.reshape(x, [-1, 1, height, width])
    x = tf.nn.conv2d_transpose(
        x, _upsample_matrix, (batch*channels, 1, height * 2, width * 2), 2, data_format='NCHW'
    )
    x = tf.reshape(x, [-1, channels, height * 2, width * 2])
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    return x


def _decodeBlock(x, shortcut, rows_odd, cols_odd, cweights, bns, activation=LeakyReLU(alpha=ALPHA)):
    #Add zero padding on bottom and right if odd dimension required at output,
    #giving an output of one greater than required
    x = ZeroPadding2D(padding=((0,rows_odd),(0,cols_odd)))(x)
    # x = UpSampling2D(size=(2,2), interpolation=UPSAMPLE_INTERP)(x)

    # up_size = np.array(x.shape)
    # up_size[1] *= 2
    # up_size[2] *= 2
    # x = bicubic_interp_2d(x,(up_size[1],up_size[2]))

    x = upsample_helper(x)
    
    #If padding was added, crop the output to match the target shape
    #print(rows_odd)
    #print(cols_odd)
    x = Cropping2D(cropping=((0,rows_odd),(0,cols_odd)))(x)

    x = Concatenate()([shortcut, x])

    x = res_Block(x, cweights, bns, activation=LeakyReLU(alpha=ALPHA))

    return x

class MuskensNet(tf.keras.layers.Layer):
  def __init__(self, channels, num_downcov_blocks):
    super(MuskensNet, self).__init__()
    self.channels = channels
    self.num_downcov_blocks = num_downcov_blocks

    weights = {}
    self.bns = {}

    # dtype = tf.float16
    weight_initializer =  tf.keras.initializers.GlorotUniform()
    k=3

    for block in range(NUM_DOWNCOV_BLOCKS):
      weights['encode_layer'+str(block)+'_'+'conv'] = []
      self.bns['encode_layer'+str(block)+'_'+'bn'] = []
      out_channels = (2**block)*HIDDEN_DIM
      if(block == 0):
        weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, self.channels, out_channels],dtype=COMPUTE_DTYPE), name='encode_layer'+str(block)+'_'+'conv1', dtype=COMPUTE_DTYPE))
      else:
        weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels//2, out_channels],dtype=COMPUTE_DTYPE), name='encode_layer'+str(block)+'_'+'conv1', dtype=COMPUTE_DTYPE))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn1'))
      weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels],dtype=COMPUTE_DTYPE), name='encode_layer'+str(block)+'_'+'conv2', dtype=COMPUTE_DTYPE))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn2'))
      weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels],dtype=COMPUTE_DTYPE), name='encode_layer'+str(block)+'_'+'conv3', dtype=COMPUTE_DTYPE))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn3'))

    for block in range(NUM_UPSAMPLING_BLOCKS):
      weights['decode_layer'+str(block)+'_'+'conv'] = []
      self.bns['decode_layer'+str(block)+'_'+'bn'] = []
      out_channels = (2**(NUM_UPSAMPLING_BLOCKS-block-1))*HIDDEN_DIM  
      in_channels = out_channels*3 # 3 is because we're concatenating the last stage of double channels

      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, in_channels, out_channels],dtype=COMPUTE_DTYPE), name='decode_layer'+str(block)+'_'+'conv1', dtype=COMPUTE_DTYPE))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn1'))
      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels],dtype=COMPUTE_DTYPE), name='decode_layer'+str(block)+'_'+'conv2', dtype=COMPUTE_DTYPE))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn2'))
      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels],dtype=COMPUTE_DTYPE), name='decode_layer'+str(block)+'_'+'conv3', dtype=COMPUTE_DTYPE))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn3'))

    weights['last_layer_conv'] = tf.Variable(weight_initializer(shape=[k, k, HIDDEN_DIM, NUM_OUT_CHANNELS], dtype=COMPUTE_DTYPE), name='last_layer_conv', dtype=COMPUTE_DTYPE)
    weights['last_layer_b'] = tf.Variable(tf.zeros([NUM_OUT_CHANNELS],dtype=COMPUTE_DTYPE), name='last_layer_b', dtype=COMPUTE_DTYPE)
    
    self.layer_weights = weights

  # @tf.function
  def call(self, inp, weights):
    print("Tracing in Muskens")
    x = inp
    blocks = [[]]*NUM_DOWNCOV_BLOCKS
    for block in range(NUM_DOWNCOV_BLOCKS):
      res_out, x = _encodeBlock(x, weights['encode_layer'+str(block)+'_'+'conv'],
                                   self.bns['encode_layer'+str(block)+'_'+'bn'])
      (_, rows, cols, _) = res_out.shape
      rows_odd = rows%2   #Boolean values. 1 if num of rows/cols is odd
      cols_odd = cols%2
      blocks[block]=[res_out, x, rows_odd, cols_odd]
      #print("pool_out.shape: ", x.shape)
    
    (x, _, _, _) = blocks[-1]
    # num_decode_blocks = len(blocks)

    for block in range(NUM_UPSAMPLING_BLOCKS):
      (shortcut, _, rows_odd, cols_odd) = blocks[NUM_UPSAMPLING_BLOCKS-1-block]
      x = _decodeBlock(x, shortcut, rows_odd, cols_odd, 
                       weights['decode_layer'+str(block)+'_'+'conv'],
                      self.bns['decode_layer'+str(block)+'_'+'bn'])
      #print("decode_out.shape:", x.shape)
    
    x = tf.nn.conv2d(input=x, filters=weights['last_layer_conv'], strides=[1,1,1,1], padding='SAME') + weights['last_layer_b']
    # x = tf.keras.layers.Activation('linear', dtype='float32')(x) # necessary for mixed precision computation
    return x

"""MAML model code"""
import sys
import tensorflow as tf
from functools import partial
## helper functions:

def accuracy(output,label):
  return tf.norm(output - label)/tf.norm(label)

# @tf.function
def my_map(*args, **kwargs):
  print("Tracing in mapping")
  return tf.map_fn(*args, **kwargs)

class MAML(tf.keras.Model):
  def __init__(self, dim_input=(50,200,1), channel=1,
               num_inner_updates=1,
               inner_update_lr=0.4, k_shot=5, learn_inner_update_lr=False):
    super(MAML, self).__init__()
    self.dim_input = (50,200,1)
    # self.dim_output = dim_output
    self.inner_update_lr = inner_update_lr
    self.loss_func = MeanAbsoluteError()
    self.channels = channel
    # self.img_size = int(np.sqrt(self.dim_input/self.channels))

    # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates
    losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
    accuracies_tr_pre, accuracies_ts = [], []

    # for each loop in the inner training loop
    outputs_ts = [[]]*num_inner_updates
    losses_ts_post = [[]]*num_inner_updates
    accuracies_ts = [[]]*num_inner_updates

    # Define the weights - these should NOT be directly modified by the
    # inner training loop
    tf.random.set_seed(seed)
    self.Unet = MuskensNet(channel, NUM_DOWNCOV_BLOCKS)

    # TO DO: update when learning the the learning rate
    self.learn_inner_update_lr = learn_inner_update_lr
    if self.learn_inner_update_lr:
      self.inner_update_lr_dict = {}
      for key in self.Unet.layer_weights.keys():
        if(isinstance(self.Unet.layer_weights[key],list)):
          self.inner_update_lr_dict[key] = [[tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d_%d' % (key, number, j), dtype=COMPUTE_DTYPE) for number in range(len(self.Unet.layer_weights[key]))] for j in range(num_inner_updates)]
        else:
          self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j), dtype=COMPUTE_DTYPE) for j in range(num_inner_updates)]

    self.accum_grad = {}
    for block in range(NUM_DOWNCOV_BLOCKS):
      self.accum_grad['encode_layer'+str(block)+'_'+'conv'] = []
      self.accum_grad['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(self.Unet.layer_weights['encode_layer'+str(block)+'_'+'conv'][0]))
      self.accum_grad['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(self.Unet.layer_weights['encode_layer'+str(block)+'_'+'conv'][1]))
      self.accum_grad['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(self.Unet.layer_weights['encode_layer'+str(block)+'_'+'conv'][2]))
    for block in range(NUM_UPSAMPLING_BLOCKS):
      self.accum_grad['decode_layer'+str(block)+'_'+'conv'] = []
      self.accum_grad['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(self.Unet.layer_weights['decode_layer'+str(block)+'_'+'conv'][0]))
      self.accum_grad['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(self.Unet.layer_weights['decode_layer'+str(block)+'_'+'conv'][1]))
      self.accum_grad['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(self.Unet.layer_weights['decode_layer'+str(block)+'_'+'conv'][2]))
    self.accum_grad['last_layer_conv'] = tf.Variable(self.Unet.layer_weights['last_layer_conv'])
    self.accum_grad['last_layer_b'] = tf.Variable(self.Unet.layer_weights['last_layer_b'])
  
  @tf.function
  def call(self, input_tr, input_ts, label_tr, label_ts, meta_batch_size=25, num_inner_updates=1):
    print("Tracing in inner")
    def task_inner_loop(inp, reuse=True,
                      meta_batch_size=25, num_inner_updates=1):
      """
        Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
        Args:
          inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
            labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
            labels used for evaluating the model after inner updates.
            Should be shapes:
              input_tr: [N*K, 200]
              input_ts: [N*K, 200]
              label_tr: [N*K, 50,200]
              label_ts: [N*K, 50,200]
        Returns:
          task_output: a list of outputs, losses and accuracies at each inner update
      """
      # the inner and outer loop data
      input_tr, input_ts, label_tr, label_ts = inp
      # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
      weights = self.Unet.layer_weights

      # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
      # evaluated on the inner loop training data
      task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

      # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
      # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
      # after i+1 inner gradient updates
      task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []
  
      #############################
      # perform num_inner_updates to get modified weights
      # modified weights should be used to evaluate performance
      # Note that at each inner update, always use input_tr and label_tr for calculating gradients
      # and use input_ts and labels for evaluating performance
      
      # new_weights = weights.copy()
      for i in range(num_inner_updates):
        with tf.GradientTape(persistent=True) as g:
          predictions = self.Unet(input_tr, weights)
          loss = self.loss_func(predictions,label_tr)
          if(i==0):
            task_output_tr_pre = predictions
            task_loss_tr_pre = loss
        gradients = g.gradient(loss, weights)
        # print(type(gradients))
        if self.learn_inner_update_lr:
          for block in range(NUM_DOWNCOV_BLOCKS):
            weights['encode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][0]*gradients['encode_layer'+str(block)+'_'+'conv'][0])
            weights['encode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][1]*gradients['encode_layer'+str(block)+'_'+'conv'][1])
            weights['encode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][2]*gradients['encode_layer'+str(block)+'_'+'conv'][2])
            if(i == 0):
              self.accum_grad['encode_layer'+str(block)+'_'+'conv'][0].assign(self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][0]*gradients['encode_layer'+str(block)+'_'+'conv'][0])
              self.accum_grad['encode_layer'+str(block)+'_'+'conv'][1].assign(self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][1]*gradients['encode_layer'+str(block)+'_'+'conv'][1])
              self.accum_grad['encode_layer'+str(block)+'_'+'conv'][2].assign(self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][2]*gradients['encode_layer'+str(block)+'_'+'conv'][2])
            else:
              self.accum_grad['encode_layer'+str(block)+'_'+'conv'][0].assign_add(self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][0]*gradients['encode_layer'+str(block)+'_'+'conv'][0])
              self.accum_grad['encode_layer'+str(block)+'_'+'conv'][1].assign_add(self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][1]*gradients['encode_layer'+str(block)+'_'+'conv'][1])
              self.accum_grad['encode_layer'+str(block)+'_'+'conv'][2].assign_add(self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][2]*gradients['encode_layer'+str(block)+'_'+'conv'][2])
          for block in range(NUM_UPSAMPLING_BLOCKS):
            weights['decode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][0]*gradients['decode_layer'+str(block)+'_'+'conv'][0])
            weights['decode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][1]*gradients['decode_layer'+str(block)+'_'+'conv'][1])
            weights['decode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][2]*gradients['decode_layer'+str(block)+'_'+'conv'][2])
            if(i == 0):
              self.accum_grad['decode_layer'+str(block)+'_'+'conv'][0].assign(self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][0]*gradients['decode_layer'+str(block)+'_'+'conv'][0])
              self.accum_grad['decode_layer'+str(block)+'_'+'conv'][1].assign(self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][1]*gradients['decode_layer'+str(block)+'_'+'conv'][1])
              self.accum_grad['decode_layer'+str(block)+'_'+'conv'][2].assign(self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][2]*gradients['decode_layer'+str(block)+'_'+'conv'][2])
            else:
              self.accum_grad['decode_layer'+str(block)+'_'+'conv'][0].assign_add(self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][0]*gradients['decode_layer'+str(block)+'_'+'conv'][0])
              self.accum_grad['decode_layer'+str(block)+'_'+'conv'][1].assign_add(self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][1]*gradients['decode_layer'+str(block)+'_'+'conv'][1])
              self.accum_grad['decode_layer'+str(block)+'_'+'conv'][2].assign_add(self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][2]*gradients['decode_layer'+str(block)+'_'+'conv'][2])
          weights['last_layer_conv'].assign_add(-self.inner_update_lr_dict['last_layer_conv'][i]*gradients['last_layer_conv'])
          weights['last_layer_b'].assign_add(-self.inner_update_lr_dict['last_layer_b'][i]*gradients['last_layer_b'])
          if(i == 0):
            self.accum_grad['last_layer_conv'].assign(self.inner_update_lr_dict['last_layer_conv'][i]*gradients['last_layer_conv'])
            self.accum_grad['last_layer_b'].assign(self.inner_update_lr_dict['last_layer_b'][i]*gradients['last_layer_b'])
          else:
            self.accum_grad['last_layer_conv'].assign_add(self.inner_update_lr_dict['last_layer_conv'][i]*gradients['last_layer_conv'])
            self.accum_grad['last_layer_b'].assign_add(self.inner_update_lr_dict['last_layer_b'][i]*gradients['last_layer_b'])
        else:
          for block in range(NUM_DOWNCOV_BLOCKS):
            # print(gradients['encode_layer'+str(block)+'_'+'conv'][0].shape, weights['encode_layer'+str(block)+'_'+'conv'][0].shape)
            weights['encode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][0])
            weights['encode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][1])
            weights['encode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][2])
          for block in range(NUM_UPSAMPLING_BLOCKS):
            weights['decode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][0])
            weights['decode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][1])
            weights['decode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][2])
          weights['last_layer_conv'].assign_add(-self.inner_update_lr*gradients['last_layer_conv'])
          weights['last_layer_b'].assign_add(-self.inner_update_lr*gradients['last_layer_b'])

        predictions_ts = self.Unet(input_ts, weights)
        task_outputs_ts.append(predictions_ts)
        loss_ts = self.loss_func(predictions_ts, label_ts)
        task_losses_ts.append(loss_ts)

      for block in range(NUM_DOWNCOV_BLOCKS):
        weights['encode_layer'+str(block)+'_'+'conv'][0].assign_add(self.accum_grad['encode_layer'+str(block)+'_'+'conv'][0])
        weights['encode_layer'+str(block)+'_'+'conv'][1].assign_add(self.accum_grad['encode_layer'+str(block)+'_'+'conv'][1])
        weights['encode_layer'+str(block)+'_'+'conv'][2].assign_add(self.accum_grad['encode_layer'+str(block)+'_'+'conv'][2])
      for block in range(NUM_UPSAMPLING_BLOCKS):
        weights['decode_layer'+str(block)+'_'+'conv'][0].assign_add(self.accum_grad['decode_layer'+str(block)+'_'+'conv'][0])
        weights['decode_layer'+str(block)+'_'+'conv'][1].assign_add(self.accum_grad['decode_layer'+str(block)+'_'+'conv'][1])
        weights['decode_layer'+str(block)+'_'+'conv'][2].assign_add(self.accum_grad['decode_layer'+str(block)+'_'+'conv'][2])
      weights['last_layer_conv'].assign_add(self.accum_grad['last_layer_conv'])
      weights['last_layer_b'].assign_add(self.accum_grad['last_layer_b'])
      #############################

      # Compute accuracies from output predictions
      task_accuracy_tr_pre = accuracy(task_output_tr_pre, label_tr)

      for j in range(num_inner_updates):
        task_accuracies_ts.append(accuracy(task_outputs_ts[j], label_ts))
      task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

      # del(task_output_tr_pre)
      # gc.collect()
      return task_output

    # input_tr, input_ts, label_tr, label_ts = inp
    # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
    unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                          False,
                          meta_batch_size,
                          num_inner_updates)
    out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
    out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
    task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
    
    result = my_map(task_inner_loop_partial, 
                       elems=(input_tr, input_ts, label_tr, label_ts),
                       dtype=out_dtype,
                       parallel_iterations=meta_batch_size)
    #result = []
    #for j in range(meta_batch_size):
    # result.append(task_inner_loop((input_tr[j], input_ts[j], label_tr[j], label_ts[j]), False, meta_batch_size, num_inner_updates))
    
    return result

"""Model training code"""
"""
Usage Instructions:
  5-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=25 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1 --logdir=logs/omniglot5way/
  20-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=16 --k_shot=1 --n_way=20 --inner_update_lr=0.1 --num_inner_updates=5 --logdir=logs/omniglot20way/
  To run evaluation, use the '--meta_train=False' flag and the '--meta_test_set=True' flag to use the meta-test set.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

# tensorboard
from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
import tensorboard
tensorboard.__version__

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = '/scratch/users/chenkaim/logs/func/%s' % stamp

writer = tf.summary.create_file_writer(logdir)

# @tf.function
def outer_train_step(input_tr, input_ts, label_tr, label_ts, model, optim, itr, meta_batch_size=25, num_inner_updates=1):
  # print("Tracing in outer")
  # tf.summary.trace_on(graph=True, profiler=True)
  with tf.GradientTape(persistent=False) as outer_tape:
    result = model(input_tr, input_ts, label_tr, label_ts, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]
    with writer.as_default():
      tf.summary.scalar('loss', total_losses_ts[-1], step=itr)
      # tf.summary.trace_export(name="my_func_trace",
      #                         step=0,
      #                         profiler_outdir=logdir)
  gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
  # print("model.trainable_variables[0][0][0][0][0]: ", model.trainable_variables[0][0][0][0][0])
  optim.apply_gradients(zip(gradients, model.trainable_variables))
  # print("optim.learning_rate = ", optim.learning_rate)

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  with writer.as_default():
      tf.summary.scalar('loss_pre', total_loss_tr_pre, step=itr)
      
  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

  return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts

def outer_eval_step(input_tr, input_ts, label_tr, label_ts, model, meta_batch_size=25, num_inner_updates=1):
  result = model(input_tr, input_ts, label_tr, label_ts, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

  outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

  return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts  


def meta_train_fn(model, exp_string, data_generator,
               n_way=5, meta_batch_size=25,
               log=True, logdir='/scratch/users/chenkaim/models/', k_shot=1, num_inner_updates=1, meta_lr=0.001, epochs=1,
               continue_train=False, continue_epoch=0):
  SUMMARY_INTERVAL = 5

  ITER_SAVE_INTERVAL = 300
  EPOCH_SAVE_INTERVAL = 5

  PRINT_INTERVAL = 5  
  TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

  meta_train_results = [[],[],[]] # iters, pre_accuracy, post_accuracy
  meta_val_results = [[],[],[]] # iters, pre_accuracy, post_accuracy

  pre_accuracies, post_accuracies = [], []

  num_classes = data_generator.num_classes

  num_data_samples = 20000
  lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(meta_lr,
                                                                decay_steps=100,
                                                                decay_rate=0.95,
                                                                staircase=False)

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
  start_epoch = continue_epoch if continue_train else 0
  for epoch in range(start_epoch, epochs):
    for itr in range(int(0.7*num_data_samples/(meta_batch_size*k_shot))):
      #############################
      #### YOUR CODE GOES HERE ####

      # sample a batch of training data and partition into
      # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
      # NOTE: The code assumes that the support and query sets have the same number of examples.
      # print("iter ", itr)
      input_meta_train, label_meta_train = data_generator.sample_batch("meta_train", meta_batch_size);
      input_tr = tf.reshape(input_meta_train[:,:,:k_shot,:,:],[input_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], model.channels])
      label_tr = tf.reshape(label_meta_train[:,:,:k_shot,:,:,:],[label_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])
      input_ts = tf.reshape(input_meta_train[:,:,k_shot:,:,:],[input_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], model.channels])
      label_ts = tf.reshape(label_meta_train[:,:,k_shot:,:,:,:],[label_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])

      #############################
      # inp = (input_tr, input_ts, label_tr, label_ts)
      
      result = outer_train_step(input_tr, input_ts, label_tr, label_ts, model, optimizer, itr, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

      if itr % SUMMARY_INTERVAL == 0:
        pre_accuracies.append(result[-2])
        post_accuracies.append(result[-1][-1])

      if (itr!=0) and itr % PRINT_INTERVAL == 0:
        print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f, train_loss: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies), result[3][-1])
        print(print_str)
        meta_train_results[0].append(itr)
        meta_train_results[1].append(np.mean(pre_accuracies))
        meta_train_results[2].append(np.mean(post_accuracies))
        
        pre_accuracies, post_accuracies = [], []

      if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
        #############################
        # sample a batch of validation data and partition it into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        input_meta_val, label_meta_val = data_generator.sample_batch("meta_val", meta_batch_size);
        input_tr = tf.reshape(input_meta_val[:,:,:k_shot,:,:], [input_meta_val.shape[0], -1, model.dim_input[0], model.dim_input[1], model.channels])
        label_tr = tf.reshape(label_meta_val[:,:,:k_shot,:,:,:], [label_meta_val.shape[0], -1, model.dim_input[0],model.dim_input[1], NUM_OUT_CHANNELS])
        input_ts = tf.reshape(input_meta_val[:,:,k_shot:,:,:], [input_meta_val.shape[0], -1, model.dim_input[0], model.dim_input[1], model.channels])
        label_ts = tf.reshape(label_meta_val[:,:,k_shot:,:,:,:], [label_meta_val.shape[0], -1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])
        print("finished sampling eval set")
        #############################

        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(input_tr, input_ts, label_tr, label_ts, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))
        meta_val_results[0].append(itr)
        meta_val_results[1].append(result[-2])
        meta_val_results[2].append(result[-1][-1])

      if(itr % ITER_SAVE_INTERVAL == 0):
        model_file = logdir + '/' + exp_string +  '/model_iter_' + str(itr)
        print("Saving to ", model_file)
        model.save_weights(model_file+'.ckpt')

    if(epoch % EPOCH_SAVE_INTERVAL == 0):
      model_file = logdir + '/' + exp_string +  '/model_epoch_' + str(epoch)
      print("Saving to ", model_file)
      model.save_weights(model_file+'.ckpt')
      #tf.saved_model.save(model, model_file)
 
  return meta_train_results, meta_val_results, model, model_file

# TO DO: change this:
NUM_META_TEST_POINTS = 20

def meta_test_fn(model, data_generator, n_way=5, meta_batch_size=25, k_shot=1,
              num_inner_updates=1):
  
  num_classes = data_generator.num_classes

  np.random.seed(1)
  random.seed(1)

  meta_test_accuracies = []

  for _iter in range(NUM_META_TEST_POINTS):
    if (_iter %5==0):
      print(_iter)
    #############################
    # sample a batch of test data and partition it into
    # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
    input_meta_test, label_meta_test = data_generator.sample_batch("meta_test", meta_batch_size);
    input_tr = tf.reshape(input_meta_test[:,:,:k_shot,:,:],[input_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], model.channels])
    label_tr = tf.reshape(label_meta_test[:,:,:k_shot,:,:,:],[label_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])
    input_ts = tf.reshape(input_meta_test[:,:,k_shot:,:,:],[input_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], model.channels])
    label_ts = tf.reshape(label_meta_test[:,:,k_shot:,:,:,:],[label_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])
    #############################
    inp = (input_tr, input_ts, label_tr, label_ts)
    result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    meta_test_accuracies.append(result[-1][-1])

  meta_test_accuracies = np.array(meta_test_accuracies)
  means = np.mean(meta_test_accuracies)
  stds = np.std(meta_test_accuracies)
  ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)

  print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
  print((means, stds, ci95))

def run_maml(n_way=5, k_shot=1, meta_batch_size=5, meta_lr=0.001,
             inner_update_lr=0.4, num_inner_updates=1,
             learn_inner_update_lr=False,
             resume=False, resume_itr=0, log=True, logdir='/scratch/users/chenkaim/models',
             data_path='/content/drive/My Drive/JonFan/data/',meta_train=True,
             meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1,
             epochs=1, continue_train=False,continue_epoch=0):


  # call data_generator and get data with k_shot*2 samples per class
  data_generator = DataGenerator(n_way, k_shot*2, n_way, k_shot*2, config={'data_folder': data_path})

  # set up MAML model
  # dim_output = data_generator.dim_output
  dim_input = (50,200,1)
  model = MAML(dim_input, channel=1,
              num_inner_updates=num_inner_updates,
              inner_update_lr=inner_update_lr,
              k_shot=k_shot,
              learn_inner_update_lr=learn_inner_update_lr)

  if meta_train_k_shot == -1:
    meta_train_k_shot = k_shot
  if meta_train_inner_update_lr == -1:
    meta_train_inner_update_lr = inner_update_lr

  exp_string = 'final_binary_5_fre_5_down_'+'cls_'+str(n_way)+'.mbs_'+str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)

  if meta_train:
    if(continue_train):
      model_file = logdir + '/' + exp_string + '/' + "model_epoch_"+str(continue_epoch-1)+".ckpt"
      print("Restoring model weights from ", model_file)
      model.load_weights(model_file)

    meta_train_results, meta_val_results,  _model, model_file = meta_train_fn(model, exp_string, data_generator,
                  n_way, meta_batch_size, log, logdir,
                  k_shot, num_inner_updates, meta_lr, epochs, continue_train, continue_epoch)
    return meta_train_results, meta_val_results,  _model, model_file
  else:
    meta_batch_size = 1

    model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
    print("Restoring model weights from ", model_file)
    model.load_weights(model_file)

    meta_test_results = meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)
    return meta_test_results

run_results = run_maml(n_way=1, k_shot=2, 
                       inner_update_lr=5e-4, num_inner_updates=2, 
                       meta_batch_size=30, 
                       epochs=31, 
                       learn_inner_update_lr=True,
                       meta_train=True,
                       continue_train=False,
                       continue_epoch=0,
                       meta_lr=1e-4,
                       data_path='/scratch/users/chenkaim/data/binary_5_fre')

from matplotlib import pyplot as plt

meta_train_results, meta_val_results,  _model, model_file = run_results



# plt.figure()
# plt.plot(meta_train_results[0],meta_train_results[1])
# plt.plot(meta_train_results[0],meta_train_results[2])
# plt.xlabel("meta_train_results")
# plt.ylabel("loss")
# plt.title("Meta-train loss")

# plt.figure()
# plt.plot(meta_val_results[0],meta_val_results[1])
# # plt.plot(meta_val_results[0],meta_val_results[2])
# plt.xlabel("meta_val_results")
# plt.ylabel("loss")
# plt.title("Meta-train loss")
# plt.show()

