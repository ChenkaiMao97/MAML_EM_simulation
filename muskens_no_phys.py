#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import pandas as pd
import keras
#from keras.models import Model
from tensorflow import keras
from tensorflow.keras.backend import int_shape
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    Cropping2D,
    Concatenate,
    BatchNormalization,
    Activation,
    LeakyReLU,
    PReLU,
    ThresholdedReLU,
    Add,
    Dropout,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K

#from initializations import he_normal

from keras.utils.generic_utils import get_custom_objects

#from phys_loss import phys_loss

#import matplotlib.pyplot as plt
#import scipy.io

#from keras.utils.vis_utils import plot_model

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6500)])

DROP_RATE = 0.2
NUM_EPOCHS = 200
ALPHA = 0.2
LEARNING_RATE = 0.00005
LR_DIV_FACTOR = 25.
PCT_START = 0.2
OMEGA = 1
BATCH_SIZE = 64
NUM_DOWNCOV_BLOCKS = 5
NUM_OUT_CHANNELS = 2
UPSAMPLE_INTERP = ["nearest", "bilinear"][1]
FNAME = '/scratch/users/chenkaim/test'
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

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

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


def _convLayer(x, num_kernels, kernel_width):
    x = Conv2D(
    filters = num_kernels
    , kernel_size = (kernel_width,kernel_width)
    , padding = 'same'
    , activation='linear'
    , kernel_initializer=initializers.he_normal(seed=None)
    , use_bias = False  #Set to false, because batch normalization is used
    )(x)
    #x = Dropout(DROP_RATE)(x)
    return x

def _convBlock(x, num_kernels, kernel_width):
    conv_layer = _convLayer(x, num_kernels, kernel_width)

    norm_layer = BatchNormalization()(conv_layer)
    #norm_layer = BatchNormalization()(conv_layer)

    x = LeakyReLU(alpha=ALPHA)(norm_layer)
    #x = ThresholdedReLU()(norm_layer)
    #x = Activation(custom_activation)(norm_layer)

    return conv_layer, norm_layer, x

def _resBlock(x, num_kernels, kernel_width):
    first_conv, _, x = _convBlock(x, num_kernels, kernel_width)
    _, _, x = _convBlock(x, num_kernels, kernel_width)
    _, x, _ = _convBlock(x, num_kernels, kernel_width)
    x = Add()([first_conv,x])

    act_out = LeakyReLU(alpha=ALPHA)(x)
    #act_out = ThresholdedReLU()(x)
    #act_out = Activation(custom_activation)(x)

    return act_out

def _encodeBlock(x, num_kernels, kernel_width):
    act_out = _resBlock(x, num_kernels, kernel_width)
    pool_out = MaxPooling2D((2, 2), strides=2)(act_out)
    return act_out, pool_out


def _decodeBlock(x, shortcut, num_kernels, rows_odd, cols_odd, kernel_width):
    #Add zero padding on bottom and right if odd dimension required at output,
    #giving an output of one greater than required
    x = ZeroPadding2D(padding=((0,rows_odd),(0,cols_odd)))(x)
    x = UpSampling2D(size=(2,2), interpolation=UPSAMPLE_INTERP)(x)
    #If padding was added, crop the output to match the target shape
    #print(rows_odd)
    #print(cols_odd)
    x = Cropping2D(cropping=((0,rows_odd),(0,cols_odd)))(x)

    x = Concatenate()([shortcut, x])

    x = _resBlock(x, num_kernels, kernel_width)

    return x

def _muskensNet(input_shape):
    input = Input(shape = input_shape)

    x = input

    blocks = []
    for block in range(NUM_DOWNCOV_BLOCKS):
        #The last block has a smaller kernel size than all other blocks
        if(block==NUM_DOWNCOV_BLOCKS-1):
            act_out, x = _encodeBlock(x, (2**block)*32, 2)
        else:
            act_out, x = _encodeBlock(x, (2**block)*32, 3)
        (_, rows, cols, _) = act_out.shape
        rows_odd = rows%2   #Boolean values. 1 if num of rows/cols is odd
        cols_odd = cols%2
        blocks.append((act_out, x, rows_odd, cols_odd))

    (x, _, _, _) = blocks.pop()
    num_decode_blocsks = len(blocks)
    for block in range(num_decode_blocsks):
        (shortcut, _, rows_odd, cols_odd) = blocks.pop()
        x = _decodeBlock(x, shortcut, (2**(num_decode_blocsks-block-1))*32, rows_odd, cols_odd, 3)

    x = _convLayer(x, NUM_OUT_CHANNELS, 3)

    model = keras.Model(inputs=[input], outputs=[x])

    return model

nn_input_dim = (100,100,1)

model = _muskensNet(nn_input_dim)

"""model.compile(loss='mae',
              optimizer=Adam(lr=LEARNING_RATE, epsilon=1e-7),
              )"""

optimizer = Adam(lr=LEARNING_RATE, epsilon=1e-7)
scheduler = LearningRateScheduler(lr_max=LEARNING_RATE, div_factor=LR_DIV_FACTOR,
                        pct_start=PCT_START)
loss_fn = MeanAbsoluteError()
model.compile(optimizer, loss_fn)


input_imgs = np.load('/home/users/chenkaim/Documents/field-predictor/input_nn3000.npy')#[0:60]
#input_metadata = np.load('input_metadata_850.npy')
targets = np.load('/home/users/chenkaim/Documents/field-predictor/target_nn3000.npy')#[0:60]

input_imgs = np.concatenate([input_imgs[:,:,:int(input_imgs.shape[2]/2)],
                            input_imgs[:,:,int(input_imgs.shape[2]/2):]], axis=1)

targets = np.concatenate([targets[:,:,:int(targets.shape[2]/2),:],
                            targets[:,:,int(targets.shape[2]/2):,:]], axis=1)

#input_imgs=np.reshape(input_imgs, (5427,100,400,1))
#np.save('input_img_1050.npy', input_imgs)

X_train, X_test, y_train, y_test = train_test_split(input_imgs, targets, test_size=0.05, random_state=42)

X_test = np.reshape(X_test, X_test.shape + (1,))
X_train = np.reshape(X_train, X_train.shape + (1,))
print(X_train.shape)
#X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=2048).batch(BATCH_SIZE)


"""model.load_weights('s50_200/no_phys.ckpt')

DEVICE_NUM = 380

out = model.predict(np.reshape(X_test[DEVICE_NUM], (1,100,100,1)))[0]

channel_1 = np.moveaxis(y_test[DEVICE_NUM], -1, 0)[0]
channel_2 = np.moveaxis(y_test[DEVICE_NUM], -1, 0)[1]

channel_1_pred = np.moveaxis(out, -1, 0)[0]
channel_2_pred = np.moveaxis(out, -1, 0)[1]

channel_1=np.concatenate([channel_1[:int(channel_1.shape[0]/2),:],
                        channel_1[int(channel_1.shape[0]/2):,:]],axis=1)
channel_2=np.concatenate([channel_2[:int(channel_2.shape[0]/2),:],
                        channel_2[int(channel_2.shape[0]/2):,:]],axis=1)
channel_1_pred=np.concatenate([channel_1_pred[:int(channel_1_pred.shape[0]/2),:],
                        channel_1_pred[int(channel_1_pred.shape[0]/2):,:]],axis=1)
channel_2_pred=np.concatenate([channel_2_pred[:int(channel_2_pred.shape[0]/2),:],
                        channel_2_pred[int(channel_2_pred.shape[0]/2):,:]],axis=1)

#device = X_test[DEVICE_NUM]

#laplace_test_arrs = channel_1
#laplace_test_arrs = np.reshape(laplace_test_arrs, (1,)+laplace_test_arrs.shape)
#laplace_test_arrs = np.append(laplace_test_arrs, np.reshape(channel_2, (1,)+channel_2.shape), axis=0)
#laplace_test_arrs = np.append(laplace_test_arrs, np.reshape(channel_1_pred, (1,)+channel_1_pred.shape), axis=0)
#laplace_test_arrs = np.append(laplace_test_arrs, np.reshape(channel_2_pred, (1,)+channel_2_pred.shape), axis=0)
#laplace_test_arrs = np.append(laplace_test_arrs, np.reshape(device, (1,)+device.shape), axis=0)
#np.save('laplace_test_arrs.npy', laplace_test_arrs)

fig, axs = plt.subplots(2,2)
img_00 = axs[0, 0].imshow(channel_1, cmap='hot', interpolation='nearest')
axs[0, 0].set_title('channel_1')
#axs[0, 0].colorbar(heatmap, shrink=0.35)
img_01 = axs[0, 1].imshow(channel_2, cmap='hot', interpolation='nearest')
axs[0, 1].set_title('channel_2')
#axs[0, 1].colorbar(heatmap, shrink=0.35)
img_10 = axs[1, 0].imshow(channel_1_pred, cmap='hot', interpolation='nearest')
axs[1, 0].set_title('channel_1-pred')
#axs[1, 0].colorbar(heatmap, shrink=0.35)
img_11 = axs[1, 1].imshow(channel_2_pred, cmap='hot', interpolation='nearest')
axs[1, 1].set_title('channel_2-pred')
fig.colorbar(img_00, ax=axs, shrink=0.35, orientation='horizontal')
#fig.colorbar(img_01, ax=axs, shrink=0.35, orientation='horizontal')
#fig.colorbar(img_10, ax=axs, shrink=0.35, orientation='horizontal')
#fig.colorbar(img_11, ax=axs, shrink=0.35, orientation='horizontal')

plt.show()
exit()"""

"""pattern = np.moveaxis(input_imgs[DEVICE_NUM], -1, 0)[0]
plt.imshow(pattern, cmap='hot')
plt.show()
exit()"""

eps_Si = 3.5674**2
reg_norm = 1

df = pd.DataFrame(columns=['epoch','train_loss', 'abs_loss', 'phys_reg', 'after_train_loss','test_loss'])

history = []
#train_loss_history = []
eval_history = []
after_eval_history = []
phys_reg_history = []
abs_loss_history = []

#total_steps = NUM_EPOCHS*len(train_dataset)

for epoch in range(NUM_EPOCHS):
    print("\nStart of epoch %d" % (epoch,),flush=True)
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            #print(x_batch_train)
            logits = model(x_batch_train, training=True)

            abs_loss = loss_fn(y_batch_train, logits)
            loss_value = abs_loss 


        """lr_step = (epoch)*len(train_dataset)+step
        new_lr = scheduler.step(lr_step/total_steps)
        K.set_value(model.optimizer.learning_rate, new_lr)"""

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 25 == 0:
            print("Seen so far: %s samples / 7927" % ((step + 1) * BATCH_SIZE),flush=True)


    history.append(loss_value.numpy())
    after_eval_history.append(model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=1))
    eval_history.append(model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1))
    abs_loss_history.append(abs_loss.numpy())
    phys_reg_history.append(0)
    #abs_loss_history.append(0)
    model.save_weights(FNAME + '.ckpt')

    df = df.append({'epoch': epoch+1, 'lr': 0,
    'train_loss': history[len(history)-1],
    'abs_loss': abs_loss_history[len(abs_loss_history)-1],
    'phys_reg': phys_reg_history[len(phys_reg_history)-1],
    'after_train_loss': after_eval_history[len(after_eval_history)-1],
    'test_loss': eval_history[len(eval_history)-1]}, ignore_index=True)

    df.to_csv(FNAME + '.csv',index=False)

history = np.array(history)
eval_history = np.array(eval_history)
print(np.reshape(history, (-1,)))
print(np.reshape(eval_history, (-1,)))


print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print('test loss, test acc:', results)
#plot_model(model, to_file='sin_muskens_model.png', show_shapes=True,show_layer_names=True)
#exit()

