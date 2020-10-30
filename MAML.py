"""Data loading scripts"""
import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import scipy.io

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
    random.shuffle(images_labels)
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
  Hr = np.array(mat['Hy_real'])
  Hi = np.array(mat['Hy_imag'])
  fields = np.stack((Hr, Hi), axis=2)

  img = np.array(mat['img'])
  
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
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Concatenate
#     Input,
#     Dense,
#     Conv2D,
#     Flatten,
#     Concatenate,
#     BatchNormalization,
#     Activation,
#     LeakyReLU,
#     PReLU,
#     ThresholdedReLU,
#     Add,
#     Dropout,
# )
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K

#from initializations import he_normal

from keras.utils.generic_utils import get_custom_objects
DROP_RATE = 0.2
NUM_EPOCHS = 200
ALPHA = 0.2
LEARNING_RATE = 0.00005
LR_DIV_FACTOR = 25.
PCT_START = 0.2
OMEGA = 1
BATCH_SIZE = 16
NUM_DOWNCOV_BLOCKS = 4
NUM_UPSAMPLING_BLOCKS = NUM_DOWNCOV_BLOCKS-1
NUM_OUT_CHANNELS = 2
UPSAMPLE_INTERP = ["nearest", "bilinear"][0]
FNAME = '/scratch/users/chenkaim/test'

seed=1234
#FNAME = 's100_100_45k/leaky_alph02_b64mae_dr0_dc4_filsx1_lkw2_v7'

def annealing_linear(start, end, pct):
    return start + pct * (end-start)

def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

class LearningRateScheduler(object):
    """
    (0, pct_start) -- linearly increase lr
    (pct_start, 1) -- cos annealing
    """
    def __init__(self, lr_max, div_factor=25., pct_start=0.2):
        super(LearningRateScheduler, self).__init__()
        self.lr_max = lr_max
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.lr_low = self.lr_max / self.div_factor

    def step(self, pct):
        # pct: [0, 1]
        if pct <= self.pct_start:
            return annealing_linear(self.lr_low, self.lr_max, pct / self.pct_start)

        else:
            return annealing_cos(self.lr_max, self.lr_low / 1e4, (
                pct - self.pct_start) / (1 - self.pct_start))

def custom_activation(x):
    return (tf.math.sin(OMEGA*tf.keras.activations.linear(x)))
#def custom_activation(x):
#    return tf.nn.leaky_relu(x, alpha=0.3)

#get_custom_objects().update({'custom_activation': Activation(custom_activation)})

class UniformRandom(tf.keras.initializers.Initializer):

    def __init__(self, omega0, is_first=True):
      self.omega0 = omega0
      self.is_first=is_first

    def __call__(self, shape, dtype=None, partition_info=None):
        scale_shape = shape
        if partition_info is not None:
            scale_shape = partition_info.full_shape
        fan_in, fan_out = _compute_fans(scale_shape)
        if(self.is_first):
            self.is_first=False
            return K.random_uniform(shape, minval=-1/fan_in, maxval=1/fan_in, dtype=dtype)
        else:
            return K.random_uniform(shape, minval=-np.sqrt(6/fan_in)/self.omega0, maxval=np.sqrt(6/fan_in)/self.omega0, dtype=dtype)

def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Args:
     shape: Integer shape tuple or TF tensor shape.
    Returns:
     A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


initializer = UniformRandom(omega0=OMEGA)

# def _convLayer(x, num_kernels, kernel_width):
#   tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME') + bweight
#     x = Conv2D(
#     filters = num_kernels
#     , kernel_size = (kernel_width,kernel_width)
#     , padding = 'same'
#     , activation='linear'
#     , kernel_initializer=initializers.he_normal(seed=None)
#     , use_bias = False  #Set to false, because batch normalization is used
#     )(x)
#     #x = Dropout(DROP_RATE)(x)
#     return x


# def _convBlock(x, num_kernels, kernel_width):
#     conv_layer = _convLayer(x, num_kernels, kernel_width)

#     norm_layer = BatchNormalization()(conv_layer)
#     #norm_layer = BatchNormalization()(conv_layer)

#     x = LeakyReLU(alpha=ALPHA)(norm_layer)
#     #x = ThresholdedReLU()(norm_layer)
#     #x = Activation(custom_activation)(norm_layer)

#     return conv_layer, norm_layer, x

def conv_block(inp, cweight, bweight, bn, activation=LeakyReLU(alpha=ALPHA)):
  """ Perform, conv, batch norm, nonlinearity, and max pool """
  stride, no_stride = [1,2,2,1], [1,1,1,1]
  conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME') + bweight
  normed = bn(conv_output)
  output = activation(normed)
  return conv_output, normed, output

# def _resBlock(x, num_kernels, kernel_width):
#     first_conv, _, x = _convBlock(x, num_kernels, kernel_width)
#     _, _, x = _convBlock(x, num_kernels, kernel_width)
#     _, x, _ = _convBlock(x, num_kernels, kernel_width)
#     x = Add()([first_conv,x])

#     act_out = LeakyReLU(alpha=ALPHA)(x)
#     #act_out = ThresholdedReLU()(x)
#     #act_out = Activation(custom_activation)(x)

#     return act_out

def res_Block(inp, cweights, bweights, bns, activation=LeakyReLU(alpha=ALPHA)):
    first_conv, _, x = conv_block(inp, cweights[0], bweights[0], bns[0], activation=LeakyReLU(alpha=ALPHA))
    _, _, x = conv_block(x, cweights[1], bweights[1], bns[1], activation=LeakyReLU(alpha=ALPHA))
    _, x, _ = conv_block(x, cweights[2], bweights[2], bns[2], activation=LeakyReLU(alpha=ALPHA))
    x = first_conv+x

    output = activation(x)

    return output

def _encodeBlock(inp, cweights, bweights, bns, activation=LeakyReLU(alpha=ALPHA)):
    res_out = res_Block(inp, cweights, bweights, bns, activation=LeakyReLU(alpha=ALPHA))
    pool_out = MaxPooling2D((2, 2), strides=2)(res_out)
    return res_out, pool_out

def _decodeBlock(x, shortcut, rows_odd, cols_odd, cweights, bweights, bns, activation=LeakyReLU(alpha=ALPHA)):
    #Add zero padding on bottom and right if odd dimension required at output,
    #giving an output of one greater than required
    x = ZeroPadding2D(padding=((0,rows_odd),(0,cols_odd)))(x)
    x = UpSampling2D(size=(2,2), interpolation=UPSAMPLE_INTERP)(x)
    #If padding was added, crop the output to match the target shape
    #print(rows_odd)
    #print(cols_odd)
    x = Cropping2D(cropping=((0,rows_odd),(0,cols_odd)))(x)

    x = Concatenate()([shortcut, x])

    x = res_Block(x, cweights, bweights, bns, activation=LeakyReLU(alpha=ALPHA))

    return x

# def _muskensNet(input_shape):
#     my_input = Input(shape = input_shape)

#     x = my_input

#     blocks = []
#     for block in range(NUM_DOWNCOV_BLOCKS):
#         #The last block has a smaller kernel size than all other blocks
#         if(block==NUM_DOWNCOV_BLOCKS-1):
#             act_out, x = _encodeBlock(x, (2**block)*32, 2)
#         else:
#             act_out, x = _encodeBlock(x, (2**block)*32, 3)
#         (_, rows, cols, _) = act_out.shape
#         rows_odd = rows%2   #Boolean values. 1 if num of rows/cols is odd
#         cols_odd = cols%2
#         blocks.append((act_out, x, rows_odd, cols_odd))

#     (x, _, _, _) = blocks.pop()
#     num_decode_blocks = len(blocks)
#     for block in range(num_decode_blocks):
#         (shortcut, _, rows_odd, cols_odd) = blocks.pop()
#         x = _decodeBlock(x, shortcut, (2**(num_decode_blocks-block-1))*32, rows_odd, cols_odd, 3)

#     x = _convLayer(x, NUM_OUT_CHANNELS, 3)

#     model = keras.Model(inputs=[my_input], outputs=[x])

#     return model

class MuskensNet(tf.keras.layers.Layer):
  def __init__(self, channels, num_downcov_blocks):
    super(MuskensNet, self).__init__()
    self.channels = channels
    self.num_downcov_blocks = num_downcov_blocks
    # self.dim_hidden = dim_hidden
    # self.dim_output = dim_output
    # self.img_size = img_size

    weights = {}
    self.bns = {}

    dtype = tf.float32
    weight_initializer =  tf.keras.initializers.GlorotUniform()
    k=3

    blocks = []
    for block in range(NUM_DOWNCOV_BLOCKS):
      weights['encode_layer'+str(block)+'_'+'conv'] = []
      weights['encode_layer'+str(block)+'_'+'b'] = []
      self.bns['encode_layer'+str(block)+'_'+'bn'] = []
      out_channels = (2**block)*32
      if(block == 0):
        weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, self.channels, out_channels]), name='encode_layer'+str(block)+'_'+'conv1', dtype=dtype))
      else:
        weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels//2, out_channels]), name='encode_layer'+str(block)+'_'+'conv1', dtype=dtype))
      weights['encode_layer'+str(block)+'_'+'b'].append(tf.Variable(tf.zeros([out_channels]), name='encode_layer'+str(block)+'_'+'b1'))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn1'))
      weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='encode_layer'+str(block)+'_'+'conv2', dtype=dtype))
      weights['encode_layer'+str(block)+'_'+'b'].append(tf.Variable(tf.zeros([out_channels]), name='encode_layer'+str(block)+'_'+'b2'))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn2'))
      weights['encode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='encode_layer'+str(block)+'_'+'conv3', dtype=dtype))
      weights['encode_layer'+str(block)+'_'+'b'].append(tf.Variable(tf.zeros([out_channels]), name='encode_layer'+str(block)+'_'+'b3'))
      self.bns['encode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='encode_layer'+str(block)+'_bn3'))

    for block in range(NUM_UPSAMPLING_BLOCKS):
      weights['decode_layer'+str(block)+'_'+'conv'] = []
      weights['decode_layer'+str(block)+'_'+'b'] = []
      self.bns['decode_layer'+str(block)+'_'+'bn'] = []
      out_channels = (2**(NUM_UPSAMPLING_BLOCKS-block-1))*32  
      in_channels = out_channels*3 # 3 is because we're concatenating the last stage of double channels

      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, in_channels, out_channels]), name='decode_layer'+str(block)+'_'+'conv1', dtype=dtype))
      weights['decode_layer'+str(block)+'_'+'b'].append(tf.Variable(tf.zeros([out_channels]), name='decode_layer'+str(block)+'_'+'b1'))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn1'))
      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='decode_layer'+str(block)+'_'+'conv2', dtype=dtype))
      weights['decode_layer'+str(block)+'_'+'b'].append(tf.Variable(tf.zeros([out_channels]), name='decode_layer'+str(block)+'_'+'b2'))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn2'))
      weights['decode_layer'+str(block)+'_'+'conv'].append(tf.Variable(weight_initializer(shape=[k, k, out_channels, out_channels]), name='decode_layer'+str(block)+'_'+'conv3', dtype=dtype))
      weights['decode_layer'+str(block)+'_'+'b'].append(tf.Variable(tf.zeros([out_channels]), name='decode_layer'+str(block)+'_'+'b3'))
      self.bns['decode_layer'+str(block)+'_'+'bn'].append(tf.keras.layers.BatchNormalization(name='decode_layer'+str(block)+'_bn3'))

    weights['last_layer_conv'] = tf.Variable(weight_initializer(shape=[k, k, 32, NUM_OUT_CHANNELS]), name='last_layer_conv', dtype=dtype)
    weights['last_layer_b'] = tf.Variable(tf.zeros([NUM_OUT_CHANNELS]), name='last_layer_b')
    
    self.layer_weights = weights

  def call(self, inp, weights):
    x = inp
    blocks = []
    for block in range(NUM_DOWNCOV_BLOCKS):
      res_out, x = _encodeBlock(x, weights['encode_layer'+str(block)+'_'+'conv'], \
                                   weights['encode_layer'+str(block)+'_'+'b'], \
                                   self.bns['encode_layer'+str(block)+'_'+'bn'])
      (_, rows, cols, _) = res_out.shape
      rows_odd = rows%2   #Boolean values. 1 if num of rows/cols is odd
      cols_odd = cols%2
      blocks.append((res_out, x, rows_odd, cols_odd))
    
    (x, _, _, _) = blocks.pop()
    # num_decode_blocks = len(blocks)

    for block in range(NUM_UPSAMPLING_BLOCKS):
      (shortcut, _, rows_odd, cols_odd) = blocks.pop()
      x = _decodeBlock(x, shortcut, rows_odd, cols_odd, \
                       weights['decode_layer'+str(block)+'_'+'conv'], \
                       weights['decode_layer'+str(block)+'_'+'b'], \
                       self.bns['decode_layer'+str(block)+'_'+'bn'])
    x = tf.nn.conv2d(input=x, filters=weights['last_layer_conv'], strides=[1,1,1,1], padding='SAME') + weights['last_layer_b']

    return x


# nn_input_dim = (100,100,1)

# model = _muskensNet(nn_input_dim)

# """model.compile(loss='mae',
#               optimizer=Adam(lr=LEARNING_RATE, epsilon=1e-7),
#               )"""

# optimizer = Adam(lr=LEARNING_RATE, epsilon=1e-7)
# scheduler = LearningRateScheduler(lr_max=LEARNING_RATE, div_factor=LR_DIV_FACTOR,
#                         pct_start=PCT_START)
# loss_fn = MeanAbsoluteError()
# model.compile(optimizer, loss_fn)


# input_imgs = np.load('/home/users/chenkaim/Documents/field-predictor/input_nn3000.npy')#[0:60]
# #input_metadata = np.load('input_metadata_850.npy')
# targets = np.load('/home/users/chenkaim/Documents/field-predictor/target_nn3000.npy')#[0:60]

# input_imgs = np.concatenate([input_imgs[:,:,:int(input_imgs.shape[2]/2)],
#                             input_imgs[:,:,int(input_imgs.shape[2]/2):]], axis=1)

# targets = np.concatenate([targets[:,:,:int(targets.shape[2]/2),:],
#                             targets[:,:,int(targets.shape[2]/2):,:]], axis=1)

# #input_imgs=np.reshape(input_imgs, (5427,100,400,1))
# #np.save('input_img_1050.npy', input_imgs)

# X_train, X_test, y_train, y_test = train_test_split(input_imgs, targets, test_size=0.05, random_state=42)

# X_test = np.reshape(X_test, X_test.shape + (1,))
# X_train = np.reshape(X_train, X_train.shape + (1,))
# print(X_train.shape)
# #X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=2048).batch(BATCH_SIZE)





"""MAML model code"""
import sys
import tensorflow as tf
from functools import partial

## helper functions:

def accuracy(output,label):
  return tf.norm(output - label)/tf.norm(label)

@tf.function
def my_map(*args, **kwargs):
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
  

  def call(self, inp, meta_batch_size=25, num_inner_updates=1):
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

      new_weights = weights.copy()
      # print("input_tr.shape", input_tr.shape)
      # print("label_tr.shape", label_tr.shape)
      task_output_tr_pre = self.Unet(input_tr, new_weights)
      task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)
      # print("task_output_tr_pre.shape", task_output_tr_pre.shape)
      with tf.GradientTape(persistent=False) as g:
        new_weights = weights.copy()
        g.watch(new_weights)
        for i in range(num_inner_updates):
          predictions = self.Unet(input_tr, new_weights)
          loss = self.loss_func(predictions,label_tr)

          gradients = g.gradient(loss, new_weights)
          print("in 1")
          if self.learn_inner_update_lr:
            for block in range(NUM_DOWNCOV_BLOCKS):
              new_weights['encode_layer'+str(block)+'_'+'conv'][0] = new_weights['encode_layer'+str(block)+'_'+'conv'][0] - self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][0]*gradients['encode_layer'+str(block)+'_'+'conv'][0]
              new_weights['encode_layer'+str(block)+'_'+'conv'][1] = new_weights['encode_layer'+str(block)+'_'+'conv'][1] - self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][1]*gradients['encode_layer'+str(block)+'_'+'conv'][1]
              new_weights['encode_layer'+str(block)+'_'+'conv'][2] = new_weights['encode_layer'+str(block)+'_'+'conv'][2] - self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'conv'][i][2]*gradients['encode_layer'+str(block)+'_'+'conv'][2]
              new_weights['encode_layer'+str(block)+'_'+'b'][0] = new_weights['encode_layer'+str(block)+'_'+'b'][0] - self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'b'][i][0]*gradients['encode_layer'+str(block)+'_'+'b'][0]
              new_weights['encode_layer'+str(block)+'_'+'b'][1] = new_weights['encode_layer'+str(block)+'_'+'b'][1] - self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'b'][i][1]*gradients['encode_layer'+str(block)+'_'+'b'][1]
              new_weights['encode_layer'+str(block)+'_'+'b'][2] = new_weights['encode_layer'+str(block)+'_'+'b'][2] - self.inner_update_lr_dict['encode_layer'+str(block)+'_'+'b'][i][2]*gradients['encode_layer'+str(block)+'_'+'b'][2]
            for block in range(NUM_UPSAMPLING_BLOCKS):
              new_weights['decode_layer'+str(block)+'_'+'conv'][0] = new_weights['decode_layer'+str(block)+'_'+'conv'][0] - self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][0]*gradients['decode_layer'+str(block)+'_'+'conv'][0]
              new_weights['decode_layer'+str(block)+'_'+'conv'][1] = new_weights['decode_layer'+str(block)+'_'+'conv'][1] - self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][1]*gradients['decode_layer'+str(block)+'_'+'conv'][1]
              new_weights['decode_layer'+str(block)+'_'+'conv'][2] = new_weights['decode_layer'+str(block)+'_'+'conv'][2] - self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'conv'][i][2]*gradients['decode_layer'+str(block)+'_'+'conv'][2]
              new_weights['decode_layer'+str(block)+'_'+'b'][0] = new_weights['decode_layer'+str(block)+'_'+'b'][0] - self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'b'][i][0]*gradients['decode_layer'+str(block)+'_'+'b'][0]
              new_weights['decode_layer'+str(block)+'_'+'b'][1] = new_weights['decode_layer'+str(block)+'_'+'b'][1] - self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'b'][i][1]*gradients['decode_layer'+str(block)+'_'+'b'][1]
              new_weights['decode_layer'+str(block)+'_'+'b'][2] = new_weights['decode_layer'+str(block)+'_'+'b'][2] - self.inner_update_lr_dict['decode_layer'+str(block)+'_'+'b'][i][2]*gradients['decode_layer'+str(block)+'_'+'b'][2]
            new_weights['last_layer_conv'] = new_weights['last_layer_conv'] - self.inner_update_lr_dict['last_layer_conv'][i]*gradients['last_layer_conv']
            new_weights['last_layer_b'] = new_weights['last_layer_b'] - self.inner_update_lr_dict['last_layer_b'][i]*gradients['last_layer_b']
          else:
            for block in range(NUM_DOWNCOV_BLOCKS):
              new_weights['encode_layer'+str(block)+'_'+'conv'][0] = new_weights['encode_layer'+str(block)+'_'+'conv'][0] - self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][0]
              new_weights['encode_layer'+str(block)+'_'+'conv'][1] = new_weights['encode_layer'+str(block)+'_'+'conv'][1] - self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][1]
              new_weights['encode_layer'+str(block)+'_'+'conv'][2] = new_weights['encode_layer'+str(block)+'_'+'conv'][2] - self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'conv'][2]
              new_weights['encode_layer'+str(block)+'_'+'b'][0] = new_weights['encode_layer'+str(block)+'_'+'b'][0] - self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'b'][0]
              new_weights['encode_layer'+str(block)+'_'+'b'][1] = new_weights['encode_layer'+str(block)+'_'+'b'][1] - self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'b'][1]
              new_weights['encode_layer'+str(block)+'_'+'b'][2] = new_weights['encode_layer'+str(block)+'_'+'b'][2] - self.inner_update_lr*gradients['encode_layer'+str(block)+'_'+'b'][2]
            for block in range(NUM_UPSAMPLING_BLOCKS):
              new_weights['decode_layer'+str(block)+'_'+'conv'][0] = new_weights['decode_layer'+str(block)+'_'+'conv'][0] - self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][0]
              new_weights['decode_layer'+str(block)+'_'+'conv'][1] = new_weights['decode_layer'+str(block)+'_'+'conv'][1] - self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][1]
              new_weights['decode_layer'+str(block)+'_'+'conv'][2] = new_weights['decode_layer'+str(block)+'_'+'conv'][2] - self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'conv'][2]
              new_weights['decode_layer'+str(block)+'_'+'b'][0] = new_weights['decode_layer'+str(block)+'_'+'b'][0] - self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'b'][0]
              new_weights['decode_layer'+str(block)+'_'+'b'][1] = new_weights['decode_layer'+str(block)+'_'+'b'][1] - self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'b'][1]
              new_weights['decode_layer'+str(block)+'_'+'b'][2] = new_weights['decode_layer'+str(block)+'_'+'b'][2] - self.inner_update_lr*gradients['decode_layer'+str(block)+'_'+'b'][2]
            new_weights['last_layer_conv'] = new_weights['last_layer_conv'] - self.inner_update_lr*gradients['last_layer_conv']
            new_weights['last_layer_b'] = new_weights['last_layer_b'] - self.inner_update_lr*gradients['last_layer_b']

          predictions_ts = self.Unet(input_ts, new_weights)
          task_outputs_ts.append(predictions_ts)
          loss_ts = self.loss_func(predictions_ts, label_ts)
          task_losses_ts.append(loss_ts)
      
      #############################

      # Compute accuracies from output predictions
      task_accuracy_tr_pre = accuracy(task_output_tr_pre, label_tr)

      for j in range(num_inner_updates):
        task_accuracies_ts.append(accuracy(task_outputs_ts[j], label_ts))
      print("in 2")
      task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

      return task_output

    input_tr, input_ts, label_tr, label_ts = inp
    # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
    unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                          False,
                          meta_batch_size,
                          num_inner_updates)
    out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
    out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
    task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
    print("111")
    result = tf.map_fn(task_inner_loop_partial,
                    elems=(input_tr, input_ts, label_tr, label_ts),
                    dtype=out_dtype,
                    parallel_iterations=meta_batch_size)
    #result = task_inner_loop((input_tr, input_ts, label_tr, label_ts),True, meta_batch_size, num_inner_updates)

    print("222")
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

def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
  with tf.GradientTape(persistent=False) as outer_tape:
    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

  gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
  optim.apply_gradients(zip(gradients, model.trainable_variables))

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

  return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts

def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
  result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

  outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

  return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts  


def meta_train_fn(model, exp_string, data_generator,
               n_way=5, meta_train_iterations=15000, meta_batch_size=25,
               log=True, logdir='/tmp/data', k_shot=1, num_inner_updates=1, meta_lr=0.001):
  SUMMARY_INTERVAL = 10
  SAVE_INTERVAL = 100
  PRINT_INTERVAL = 10  
  TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

  meta_train_results = [[],[],[]] # iters, pre_accuracy, post_accuracy
  meta_val_results = [[],[]] # iters, val_accuracy

  pre_accuracies, post_accuracies = [], []

  num_classes = data_generator.num_classes

  optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

  for itr in range(meta_train_iterations):
    #############################
    #### YOUR CODE GOES HERE ####

    # sample a batch of training data and partition into
    # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
    # NOTE: The code assumes that the support and query sets have the same number of examples.

    input_meta_train, label_meta_train = data_generator.sample_batch("meta_train", meta_batch_size);
    input_tr = tf.reshape(input_meta_train[:,:,:k_shot,:,:],[input_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], model.channels])
    label_tr = tf.reshape(label_meta_train[:,:,:k_shot,:,:,:],[label_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])
    input_ts = tf.reshape(input_meta_train[:,:,k_shot:,:,:],[input_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], model.channels])
    label_ts = tf.reshape(label_meta_train[:,:,k_shot:,:,:,:],[label_meta_train.shape[0],-1, model.dim_input[0], model.dim_input[1], NUM_OUT_CHANNELS])

    #############################
    print("finished sampling training set")
    inp = (input_tr, input_ts, label_tr, label_ts)
    
    result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

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
      result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

      print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))
      meta_val_results[0].append(itr)
      meta_val_results[1].append(result[-1][-1])

  model_file = logdir + '/' + exp_string +  '/model' + str(itr)
  print("Saving to ", model_file)
  model.save_weights(model_file)
 
  return meta_train_results, meta_val_results

# TO DO: change this:
NUM_META_TEST_POINTS = 600

def meta_test_fn(model, data_generator, n_way=5, meta_batch_size=25, k_shot=1,
              num_inner_updates=1):
  
  num_classes = data_generator.num_classes

  np.random.seed(1)
  random.seed(1)

  meta_test_accuracies = []

  for _ in range(NUM_META_TEST_POINTS):
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
             resume=False, resume_itr=0, log=True, logdir='/tmp/data',
             data_path='/scratch/users/chenkaim/data/',meta_train=True,
             meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1):


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

  exp_string = 'cls_'+str(n_way)+'.mbs_'+str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)

  if meta_train:
    meta_train_results, meta_val_results = meta_train_fn(model, exp_string, data_generator,
                  n_way, meta_train_iterations, meta_batch_size, log, logdir,
                  k_shot, num_inner_updates, meta_lr)
    return meta_train_results, meta_val_results
  else:
    meta_batch_size = 1

    model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
    print("Restoring model weights from ", model_file)
    model.load_weights(model_file)

    meta_test_results = meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)
    return meta_test_results
  
run_results = run_maml(n_way=1, k_shot=5, inner_update_lr=4.0, num_inner_updates=1,meta_train_iterations=200, learn_inner_update_lr=False)
