"""Data loading scripts"""
import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import scipy.io
import gc

from datetime import datetime

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

    random.seed(123)
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
      mat_paths = get_mat_paths(sampled_data_folders, n_samples=num_samples_per_class, shuffle=shuffle)
      print(mat_paths[0])
      images_and_labels = [mat_to_input_and_field(
        li) for li in mat_paths]
      images = np.array([i[0] for i in images_and_labels])
      labels = np.array([i[1] for i in images_and_labels])
      # labels = np.array(labels).astype(np.int32)
      labels = np.reshape(labels, (num_classes, num_samples_per_class, self.img_size[0], self.img_size[1], NUM_OUT_CHANNELS))
      images = np.reshape(images, (num_classes, num_samples_per_class, self.img_size[0], self.img_size[1],1))
      #TO DO: rewrite shuffle using zip
      # batch = np.concatenate([labels, images], 2)
      # if shuffle:
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
NUM_DOWNCOV_BLOCKS = 4
NUM_UPSAMPLING_BLOCKS = NUM_DOWNCOV_BLOCKS-1
NUM_OUT_CHANNELS = 2
HIDDEN_DIM = 32
UPSAMPLE_INTERP = ["nearest", "bilinear"][1]
FNAME = '/scratch/users/chenkaim/test'

seed=1234
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

_upsample_matrix: tf.Tensor = tf.ones([2, 2, 1, 1])
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

    dtype = tf.float32
    weight_initializer =  tf.keras.initializers.GlorotUniform()
    k=3

    for block in range(NUM_DOWNCOV_BLOCKS):
      weights['encode_layer'+str(block)+'_'+'conv'] = []
      self.bns['encode_layer'+str(block)+'_'+'bn'] = []
      out_channels = (2**block)*HIDDEN_DIM
      if(block == 0):
        weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, self.channels, out_channels]), name='encode_layer'+str(block)+'_'+'conv1', dtype=dtype))
      else:
        weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels//2, out_channels]), name='encode_layer'+str(block)+'_'+'conv1', dtype=dtype))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn1'))
      weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='encode_layer'+str(block)+'_'+'conv2', dtype=dtype))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn2'))
      weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='encode_layer'+str(block)+'_'+'conv3', dtype=dtype))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn3'))

    for block in range(NUM_UPSAMPLING_BLOCKS):
      weights['decode_layer'+str(block)+'_'+'conv'] = []
      self.bns['decode_layer'+str(block)+'_'+'bn'] = []
      out_channels = (2**(NUM_UPSAMPLING_BLOCKS-block-1))*HIDDEN_DIM  
      in_channels = out_channels*3 # 3 is because we're concatenating the last stage of double channels

      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, in_channels, out_channels]), name='decode_layer'+str(block)+'_'+'conv1', dtype=dtype))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn1'))
      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='decode_layer'+str(block)+'_'+'conv2', dtype=dtype))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn2'))
      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='decode_layer'+str(block)+'_'+'conv3', dtype=dtype))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn3'))

    weights['last_layer_conv'] = tf.Variable(weight_initializer(shape=[k, k, HIDDEN_DIM, NUM_OUT_CHANNELS]), name='last_layer_conv', dtype=dtype)
    weights['last_layer_b'] = tf.Variable(tf.zeros([NUM_OUT_CHANNELS]), name='last_layer_b')
    
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

    return x

"""MAML model code"""
import sys
import tensorflow as tf
from functools import partial

## helper functions:

def accuracy(output,label):
  return tf.norm(output - label)/tf.norm(label)

@tf.function
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
        if(type(self.Unet.layer_weights[key]) is list):
          self.inner_update_lr_dict[key] = [[tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d_%d' % (key, number, j)) for number in range(len(self.Unet.layer_weights[key]))] for j in range(num_inner_updates)]
        else:
          self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in range(num_inner_updates)]
  
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

      temp_weights = weights.copy()
      # print("input_tr.shape", input_tr.shape)
      # print("label_tr.shape", label_tr.shape)
      task_output_tr_pre = self.Unet(input_tr, temp_weights)
      task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)
      # print("task_output_tr_pre.shape", task_output_tr_pre.shape)
      new_weights = weights.copy()
      for i in range(num_inner_updates):
        with tf.GradientTape(persistent=True) as g:
          g.watch(new_weights)
          predictions = self.Unet(input_tr, new_weights)
          loss = self.loss_func(predictions,label_tr)
          # del(predictions)
          # gc.collect()
          # print(predictions.shape, label_tr.shape)
        gradients = g.gradient(loss, new_weights)
        # print(type(gradients))
        if self.learn_inner_update_lr:
          for block in range(NUM_DOWNCOV_BLOCKS):
            new_weights['encode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][0]*gradients['encode_layer'+str(block)+'_'+'conv'][0])
            new_weights['encode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][1]*gradients['encode_layer'+str(block)+'_'+'conv'][1])
            new_weights['encode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][2]*gradients['encode_layer'+str(block)+'_'+'conv'][2])
            # new_weights['encode_layer'+str(block)+'_'+'b'][0].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'b'][i][0]*gradients['encode_layer'+str(block)+'_'+'b'][0])
            # new_weights['encode_layer'+str(block)+'_'+'b'][1].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'b'][i][1]*gradients['encode_layer'+str(block)+'_'+'b'][1])
            # new_weights['encode_layer'+str(block)+'_'+'b'][2].assign_add(-self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'b'][i][2]*gradients['encode_layer'+str(block)+'_'+'b'][2])
          for block in range(NUM_UPSAMPLING_BLOCKS):
            new_weights['decode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][0]*gradients['decode_layer'+str(block)+'_'+'conv'][0])
            new_weights['decode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][1]*gradients['decode_layer'+str(block)+'_'+'conv'][1])
            new_weights['decode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][2]*gradients['decode_layer'+str(block)+'_'+'conv'][2])
            # new_weights['decode_layer'+str(block)+'_'+'b'][0].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'b'][i][0]*gradients['decode_layer'+str(block)+'_'+'b'][0])
            # new_weights['decode_layer'+str(block)+'_'+'b'][1].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'b'][i][1]*gradients['decode_layer'+str(block)+'_'+'b'][1])
            # new_weights['decode_layer'+str(block)+'_'+'b'][2].assign_add(-self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'b'][i][2]*gradients['decode_layer'+str(block)+'_'+'b'][2])
          new_weights['last_layer_conv'].assign_add(-self.inner_update_lr_dict['last_layer_conv'][i]*gradients['last_layer_conv'])
          new_weights['last_layer_b'].assign_add(-self.inner_update_lr_dict['last_layer_b'][i]*gradients['last_layer_b'])
        else:
          for block in range(NUM_DOWNCOV_BLOCKS):
            new_weights['encode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][0])
            new_weights['encode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][1])
            new_weights['encode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][2])
            # new_weights['encode_layer'+str(block)+'_'+'b'][0].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'b'][0])
            # new_weights['encode_layer'+str(block)+'_'+'b'][1].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'b'][1])
            # new_weights['encode_layer'+str(block)+'_'+'b'][2].assign_add(-self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'b'][2])
          for block in range(NUM_UPSAMPLING_BLOCKS):
            new_weights['decode_layer'+str(block)+'_'+'conv'][0].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][0])
            new_weights['decode_layer'+str(block)+'_'+'conv'][1].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][1])
            new_weights['decode_layer'+str(block)+'_'+'conv'][2].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][2])
            # new_weights['decode_layer'+str(block)+'_'+'b'][0].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'b'][0])
            # new_weights['decode_layer'+str(block)+'_'+'b'][1].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'b'][1])
            # new_weights['decode_layer'+str(block)+'_'+'b'][2].assign_add(-self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'b'][2])
          new_weights['last_layer_conv'].assign_add(-self.inner_update_lr*gradients['last_layer_conv'])
          new_weights['last_layer_b'].assign_add(-self.inner_update_lr*gradients['last_layer_b'])
          
        predictions_ts = self.Unet(input_ts, new_weights)
        task_outputs_ts.append(predictions_ts)
        loss_ts = self.loss_func(predictions_ts, label_ts)
        task_losses_ts.append(loss_ts)
      #     del(predictions_ts)
      #     gc.collect()
      # del g
          
      # new_weights.clear()
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

from PIL import Image
from matplotlib import cm

def visualize_maml(n_way=5, k_shot=1, meta_batch_size=5, meta_lr=0.001,
             inner_update_lr=0.4, num_inner_updates=1,
             learn_inner_update_lr=False,
             resume=False, resume_itr=0, log=True, logdir='/scratch/users/chenkaim/models',
             data_path='/content/drive/My Drive/JonFan/data/',
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

  #exp_string = '5_fre'+'cls_'+str(n_way)+'.mbs_'+str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)
  exp_string = 'cls_1.mbs_1.k_shot_32.inner_numstep_1.inner_updatelr_0.004.learn_inner_update_lr_False'

  model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
  print("Restoring model weights from ", model_file)
  model.load_weights(model_file)

    # meta_train_results, meta_val_results,  _model, model_file = meta_train_fn(model, exp_string, data_generator,
    #               n_way, meta_batch_size, log, logdir,
    #               k_shot, num_inner_updates, meta_lr, epochs, continue_train, continue_epoch)
    # return meta_train_results, meta_val_results,  _model, model_file
  for i in range(1): # show 10 random results
    data_generator.sample_batch("meta_val", meta_batch_size);
    input_meta_test, label_meta_test = data_generator.sample_batch("meta_test", meta_batch_size);
    input_tr = tf.reshape(input_meta_test[:,:,:k_shot,:,:],[input_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], model.channels])
    label_tr = tf.reshape(label_meta_test[:,:,:k_shot,:,:,:],[label_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])
    input_ts = tf.reshape(input_meta_test[:,:,k_shot:,:,:],[input_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], model.channels])
    label_ts = tf.reshape(label_meta_test[:,:,k_shot:,:,:,:],[label_meta_test.shape[0], -1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])
    #############################
    inp = (input_tr, input_ts, label_tr, label_ts)
    result = model(inp[0],inp[1],inp[2],inp[3], meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    print("len(output_ts), outputs_ts.shape: ", len(outputs_ts), outputs_ts[0].shape)
    print("label_ts.shape: ", label_ts.shape)

    for j in range(1):
      for k in range(1):
        out1 = outputs_tr[i,k,:,:,0].numpy()
        scale = np.amax(np.abs(out1))
        out1 = (out1/scale + 1)/2
        img1 = Image.fromarray(np.uint8(cm.seismic(out1)*255))
        img1.save('Hy_real_pre_'+str(i)+'_'+str(k)+'.png')
        img1.show()

        out2 = outputs_tr[i,k,:,:,1].numpy()
        scale = np.amax(np.abs(out2))
        out2 = (out2/scale + 1)/2
        img2 = Image.fromarray(np.uint8(cm.seismic(out2)*255))
        img2.save('Hy_imag_pre_'+str(i)+'_'+str(k)+'.png')
        img2.show()

        out3 = label_tr[i,k,:,:,0].numpy()
        scale = np.amax(np.abs(out3))
        out3 = (out3/scale + 1)/2
        img3 = Image.fromarray(np.uint8(cm.seismic(out3)*255))
        img3.save('Hy_real_pre_lb_'+str(i)+'_'+str(k)+'.png')
        img3.show()

        out4 = label_tr[i,k,:,:,1].numpy()
        scale = np.amax(np.abs(out4))
        out4 = (out4/scale + 1)/2
        img4 = Image.fromarray(np.uint8(cm.seismic(out4)*255))
        img4.save('Hy_imag_pre_lb_'+str(i)+'_'+str(k)+'.png')
        img4.show()

        out1 = outputs_ts[0][i,k,:,:,0].numpy()
        scale = np.amax(np.abs(out1))
        out1 = (out1/scale + 1)/2
        img1 = Image.fromarray(np.uint8(cm.seismic(out1)*255))
        img1.save('Hy_real_after_'+str(i)+'_'+str(k)+'.png')
        img1.show()

        out2 = outputs_ts[0][i,k,:,:,1].numpy()
        scale = np.amax(np.abs(out2))
        out2 = (out2/scale + 1)/2
        img2 = Image.fromarray(np.uint8(cm.seismic(out2)*255))
        img2.save('Hy_imag_after_'+str(i)+'_'+str(k)+'.png')
        img2.show()

        out3 = label_ts[i,k,:,:,0].numpy()
        scale = np.amax(np.abs(out3))
        out3 = (out3/scale + 1)/2
        img3 = Image.fromarray(np.uint8(cm.seismic(out3)*255))
        img3.save('Hy_real_after_lb_'+str(i)+'_'+str(k)+'.png')
        img3.show()

        out4 = label_ts[i,k,:,:,1].numpy()
        scale = np.amax(np.abs(out4))
        out4 = (out4/scale + 1)/2
        img4 = Image.fromarray(np.uint8(cm.seismic(out4)*255))
        img4.save('Hy_imag_after_lb_'+str(i)+'_'+str(k)+'.png')
        img4.show()


run_results = visualize_maml(n_way=1, k_shot=8, 
                       inner_update_lr=.04, num_inner_updates=1, 
                       meta_batch_size=4, 
                       epochs=16, 
                       learn_inner_update_lr=False,
                       continue_train=True,
                       continue_epoch=11,
                       data_path='/scratch/users/chenkaim/data/binary_1000')

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

