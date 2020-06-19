# -*- coding: utf-8 -*-
import tensorflow as tf
assert tf.test.is_gpu_available()

"""# Experiment Notes

- 1 vs 2.5 upper bound (1 seems to work fine)
- Deeper vs more filters
  - 8 as lowest with 32 filters ( ~750k parameters) gave good results around 900 epochs for overfitting test of 3. Didn't generate all 3 equally.
    - It was able to generate some equal values for 10 samples!
  - 4 as lowest with 32 filters ( ~2.9M parameters) gave good (similar to 8 as lowest with 32 filters) for overfitting test of 3. Didn't generate all 3 equally.
  - 8 as lowest with 64 filters (~2.9M parameters) gave overfitting results generating only the frog again... although from some ofthe samples in the middle (500 epochs ish) it generates 3.

# What about a GAN + Self correcting U-Net ? That would make for a cool architecture
# Following CGAN -> adding a 1-hot vector encoding of the label to the training data
# Simulated Annealing?
# Generator -> VAE -> Discriminator?
# What about feeding in a dicriminator's confidence level as a temperature during the autoregressive? Inverse confidence?
# What about a 3 dimensional GAN?
# What about adding attention to the model?

# Umut Notes
- Add a stop condition to the softmax
    - Tried both 2 outputs and just an extra variable to the softmax
    - 2 outputs fails due to it having too much weight to the loss and the loss fluctuates like crazy
    - extra variable fails as the probability is still small even for an original image. Not sure why. Maybe because each time wew generate we use a new random which causes the dataset to be imbalanced?
- 2 steps process (pick note and then choose how much through binary cross entropy)
"""

import mnist
import scipy.misc
from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from keras.datasets import cifar10


def make_grayscale(data, dtype=np.float32):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    rst = rst.astype(np.uint8)
    return rst

def create_image(image, name, image_shape=(32, 32), is_grayscale=True):
    img_arr = deepcopy(image.reshape(image_shape)).astype(np.uint8)
    img = Image.fromarray(img_arr.astype(np.uint8), 'L')
    # pprint(img_arr)
    # print("img shape: {}. img sum: {}".format(img_arr.shape, img_arr.sum()))
    img.save(name)
    return img

is_single = False
is_grayscale = True
is_cifar_10 = True

n_filters_start=32
num_sub_layers = 2
learning_rate = 0.001
is_leaky_relu = False
is_batch_norm = True

if is_single:
    num_samples = 2
    epochs_per_sample = 1000
else:
    num_samples = 60000
    epochs_per_sample = 100


if is_cifar_10:
    (images, labels), (_, _) = cifar10.load_data()
    images = make_grayscale(images)
else:
    images = mnist.train_images()

    # np.random.shuffle(images)
images = images[:num_samples, :, :]


if not is_grayscale:
    # For black and white
    images[images > 0] = 1
    # images = images / 255.0

# pprint(images)
print(images.shape)

# labels = mnist.train_labels()
# n_labels = np.max(labels) + 1
# labels = np.eye(n_labels)[labels]
# print(labels.shape)

if is_cifar_10:
    image_shape = images[0].shape
else:
    image_shape = np.expand_dims(images[0], axis=-1).shape 

print(images[0])
create_image(images[0], 'my.png', image_shape=image_shape[:-1], is_grayscale=is_grayscale)
create_image(images[-1], 'my2.png', image_shape=image_shape[:-1], is_grayscale=is_grayscale)
print(image_shape)

def to_one_hot(arr):
    arr = deepcopy(arr)
    arr = arr.astype(np.uint8)
    n_values = 256
    one_hot = np.eye(n_values)[arr]
    one_hot = one_hot.astype(np.uint8)
    return one_hot

one_hot = to_one_hot(images[0])
print(one_hot.shape)

argmax_res = np.argmax(one_hot, axis=-1)
print(argmax_res)

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Flatten, Dense, Softmax, Reshape, Activation, LeakyReLU, ReLU
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.nn import softmax_cross_entropy_with_logits

def built_in_softmax_kl_loss(target, output):
    target = K.flatten(target)
    output = K.flatten(output)
    
    target = target / K.sum(target)
    output = K.softmax(output)
    return keras.losses.kullback_leibler_divergence(target, output)

def intensity_softmax_loss(target, output):
    return keras.losses.categorical_crossentropy(target, output, from_logits=True)

keras.losses.built_in_softmax_kl_loss = built_in_softmax_kl_loss
keras.losses.intensity_softmax_loss = intensity_softmax_loss

def conv_layer(n_filters, filter_size, conv):
    for _ in range(3):
        conv = Conv2D(n_filters, filter_size, padding='same')(conv)
        if is_batch_norm:
            conv = BatchNormalization()(conv)
        if is_leaky_relu:
            conv = LeakyReLU()(conv)
        else:
            conv = ReLU()(conv)
    return conv    
 
 
def unet_model(input_size, n_filters_start, growth_factor=2,
               upconv=False, is_grayscale=True, num_sub_layers=2, learning_rate=0.001):
    droprate=0.5
    n_filters = n_filters_start
    inputs = Input(input_size)
    conv_first = conv_layer(n_filters, (3, 3), inputs)
    pool_first = MaxPooling2D(pool_size=(2, 2))(conv_first)

    prev_pool = pool_first
    hidden_layers = []
    for _ in range(num_sub_layers):
        n_filters *= growth_factor
        conv = conv_layer(n_filters, (3, 3), prev_pool)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        pool = Dropout(droprate)(pool)
        prev_pool = pool
        hidden_layers.append(conv)
 
    n_filters *= growth_factor
    conv_mid = conv_layer(n_filters, (3, 3), prev_pool)
    # print(hidden_layers)
 
    n_filters //= growth_factor
    if upconv:
        up_first = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv_mid), hidden_layers[-1]])
    else:
        up_first = concatenate([UpSampling2D(size=(2, 2))(conv_mid), hidden_layers[-1]])
    conv_mid_2 = conv_layer(n_filters, (3, 3), up_first)
    conv_mid_2 = Dropout(droprate)(conv_mid_2)

    prev_conv = conv_mid_2
    for i in range(num_sub_layers - 1):
        n_filters //= growth_factor
        if upconv:
            up = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(prev_conv), hidden_layers[-i-2]])
        else:
            up = concatenate([UpSampling2D(size=(2, 2))(prev_conv), hidden_layers[-i-2]])
        conv = conv_layer(n_filters, (3, 3), up)
        conv = Dropout(droprate)(conv)
        prev_conv = conv
 
    n_filters //= growth_factor
    if upconv:
        up_last = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(prev_conv), conv_first])
    else:
        up_last = concatenate([UpSampling2D(size=(2, 2))(prev_conv), conv_first])
    conv_last = conv_layer(n_filters, (3, 3), up_last)
 
    softmax_out = Conv2D(1, 1, activation='linear', name='softmax_out')(conv_last)

    if is_grayscale:
        sigmoid_out = Conv2D(256, 1, padding='valid', name='intensity_conv')(conv_last)
        sigmoid_out = Reshape((*image_shape[:-1], 256))(sigmoid_out)
        # sigmoid_out = Activation('softmax')(sigmoid_out)
        model = Model(inputs=inputs, outputs=[softmax_out, sigmoid_out])
        model.compile(optimizer=Adam(lr=learning_rate), loss=[built_in_softmax_kl_loss, intensity_softmax_loss], metrics=['categorical_accuracy'])
    else:
        intensity_softmax = Conv2D(256 * 3, 1, padding='valid', name='intensity_conv')(conv_last)
        intensity_softmax = Reshape((*image_shape, 256))(intensity_softmax)
        # intensity_softmax = Activation('softmax', name='intensity_softmax')(intensity_softmax)
        model = Model(inputs=inputs, outputs=[softmax_out, intensity_softmax])
        model.compile(optimizer=Adam(lr=learning_rate), loss=[built_in_softmax_kl_loss, intensity_softmax_loss], metrics=['categorical_accuracy'])

    # model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = unet_model(input_size=image_shape, n_filters_start=n_filters_start, is_grayscale=is_grayscale, num_sub_layers=num_sub_layers, learning_rate=learning_rate)



# discriminator_model = discriminator(input_size=image_shape)

from copy import deepcopy
import math
import itertools
import time
import random

noise_upper_bound = 2.5

def mask_image_with_noise(image, is_grayscale=True):
    image = deepcopy(image)
    sampling_percentage_mask = np.random.uniform(0, 100)
    sampling_percentage_noise = np.random.uniform(0, noise_upper_bound)
    non_zero = np.nonzero(image)
    mask = np.full(len(non_zero[0]), False)
    noise = np.full(len(non_zero[0]), False)
    amount_to_mask = math.floor(len(non_zero[0]) * (sampling_percentage_mask / 100.0))
    mask[:amount_to_mask] = True
    amount_of_noise = math.floor(len(non_zero[0]) * (sampling_percentage_noise / 100.0))
    noise[:amount_of_noise] = True
    np.random.shuffle(mask)
    np.random.shuffle(noise)

    output_image = deepcopy(image)
    xor_target = np.full(output_image.shape, False)

    r1 = list(itertools.compress(non_zero[0], mask))
    c1 = list(itertools.compress(non_zero[1], mask))
    output_image[r1, c1] = 0
    xor_target[r1, c1] = True

    # There might be overlap but that is ok
    r2 = list(itertools.compress(non_zero[0], noise))
    c2 = list(itertools.compress(non_zero[1], noise))
    random_values = np.random.uniform(0, 1, (len(r2), 1))
    random_values *= 255
    random_values = np.around(random_values)
    random_values = random_values.astype(np.uint8)
    output_image[r2, c2] = random_values
    xor_target[r2, c2] = True

    return output_image, xor_target


class ImageGenerator(keras.utils.Sequence):
    def __init__(self, sample_list, image_shape, batch_size, samples_per_data_item, stops_per_data_item, is_grayscale=True, seed=None):
        print("sample_list: {}".format(len(sample_list)))
        self.sample_list = sample_list
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.samples_per_data_item = samples_per_data_item
        self.stops_per_data_item = stops_per_data_item
        self.is_grayscale = is_grayscale
        self.sample_index = 0
        self.seed = seed
        self.dtype = np.uint8
        # if self.seed is not None:
        #     np.random.seed(self.seed)

    def generate_training_pairs(self):
        '''
        Generates Training Pairs till @training_input / @training_target have @batch_size files.
        '''
        training_input = []
        training_original = []
        training_target = []
        while len(training_input) < self.batch_size:
            original_image = deepcopy(self.sample_list[self.sample_index])
            original_image = original_image.reshape(self.image_shape)
            binary_image = deepcopy(original_image)
            binary_image[binary_image > 0] = 1
            self.sample_index = (self.sample_index + 1) % len(self.sample_list)
            # print("sample_list length: {}. sample_index: {}".format(
            #     len(self.sample_list), self.sample_index))
            try:
                # augment by adding and removing random values in the array

                # Add random values
                for _ in range(self.samples_per_data_item):
                    original_image = original_image.astype(self.dtype)
                    input_image, xor_target = mask_image_with_noise(original_image, is_grayscale=self.is_grayscale)

                    input_image = input_image.astype(self.dtype)
                    xor_target = xor_target.astype(self.dtype)

                    training_input.append(deepcopy(input_image))
                    training_original.append(to_one_hot(np.squeeze(original_image)))
                    # training_original.append(deepcopy(original_image))
                    training_target.append(deepcopy(xor_target))

            except Exception as e:
                print('Error generating input and target pair')
                traceback.print_exc()
        training_input = np.asarray(training_input)
        training_target = np.asarray(training_target)
        training_original = np.asarray(training_original)
        return training_input, training_target, training_original

    def save_image(self, img_arr, img_name):
        # img_arr = img_arr.reshape(self.image_shape)
        print(img_name)
        print("img shape: {}. img sum: {}".format(img_arr.shape, img_arr.sum()))
        img_arr = img_arr[:, :, 0]
        print("img shape: {}. img sum: {}".format(img_arr.shape, img_arr.sum()))
        print(img_arr)
        #pprint(img_arr)
        print("img shape: {}. img sum: {}".format(img_arr.shape, img_arr.sum()))
        img = Image.fromarray(img_arr.astype(np.uint8), 'L')
        img.save(img_name)

    def get_random_training_pair(self):
        training_input, training_target, training_original = self.generate_training_pairs()
        print("training_input shape: {}".format(training_input.shape))
        index = random.randrange(0, len(training_input))
        self.save_image(deepcopy(training_input[index]), 'training_input.png')
        self.save_image(deepcopy(training_target[index]) * 255, 'training_target.png')
        print(training_original.shape)
        original_image = deepcopy(training_original[index])
        original_image = np.argmax(original_image, axis=-1)
        print(original_image.shape)
        original_image = np.expand_dims(original_image, axis=-1)
        self.save_image(original_image, 'training_original.png')

    def generate_validation_samples(self):
        old_batch_size = self.batch_size
        self.batch_size = len(self.sample_list) * (self.samples_per_data_item + self.stops_per_data_item)
        training_input, training_target, training_original = self.generate_training_pairs()
        # training_input = np.asarray(self.training_input[:self.batch_size])
        # training_target = np.asarray(self.training_target[:self.batch_size])
        self.batch_size = old_batch_size
        if self.is_grayscale:
            return training_input, [training_target, training_original]
        else:
            return training_input, training_target

    def __getitem__(self, index):
        '''Generates 1 batch of data'''
        training_input, training_target, training_original = self.generate_training_pairs()
        # training_input = np.asarray(self.training_input[:self.batch_size])
        # training_target = np.asarray(self.training_target[:self.batch_size])
        # self.training_input = self.training_input[self.batch_size:]
        # self.training_target = self.training_target[self.batch_size:]
        # print("training input sum: {}. target sum: {}".format(training_input.sum(), training_target.sum()))
        if is_grayscale:
            return training_input, [training_target, training_original]
        else:
            return np.asarray(training_input), np.asarray(training_target)

    def __len__(self):
        '''Number of batches / epoch'''
        # print("sample_list: {}. samples_per_data_item: {}, batch size: {}".
        #       format(len(self.sample_list), self.samples_per_data_item,
        #              self.batch_size))
        samples_to_generate = int(
            (len(self.sample_list) * (self.samples_per_data_item + self.stops_per_data_item)) /
            self.batch_size)
        # print("samples to generate: {}".format(samples_to_generate))
        return samples_to_generate
    
    # def on_epoch_begin(self):
    #     if self.seed is not None:
    #         np.random.seed(self.seed)
    #     else:
    #         np.random.seed(time.time())

# Config
stops_per_data_item = 0
if is_single:
    batch_size = num_samples * 32
    samples_per_data_item =   1 * 32
    split = 1
else:
    batch_size = 64
    samples_per_data_item = 1
    split = 0.9

training_samples = images[:int(len(images) * split)]
validation_samples = images[int(len(images) * split):]

print("training samples: {}. validation samples: {}".format(len(training_samples), len(validation_samples)))

steps_per_epoch = int(len(training_samples) * (samples_per_data_item + stops_per_data_item) / batch_size)
print("steps per epoch: {}".format(steps_per_epoch))

training_generator = ImageGenerator(
    sample_list=training_samples,
    image_shape=image_shape,
    batch_size=batch_size,
    samples_per_data_item=samples_per_data_item,
    stops_per_data_item=stops_per_data_item,
    is_grayscale=is_grayscale)

validation_generator = ImageGenerator(
    sample_list=validation_samples,
    image_shape=image_shape,
    batch_size=batch_size,
    samples_per_data_item=samples_per_data_item,
    stops_per_data_item=stops_per_data_item,
    is_grayscale=is_grayscale)

validation_data = validation_generator.generate_validation_samples()

# print("validation data input and target shape: {}".format(validation_data[0].shape))

training_generator.get_random_training_pair()

if is_single:
    is_single_text = "single"
else:
    is_single_text = "full"

model_custom_name = 'cifar-grayscale'
model_full_name = '{}-num-samples-{}-noise-upper-{}-num-sub-layers-{}-mini-batch-{}-samples-per-item-{}-lr-{}-is-leaky-{}-is-batch-norm-{}-n_filters-start-{}-{}'.format(model_custom_name, num_samples, noise_upper_bound, num_sub_layers, batch_size, samples_per_data_item, learning_rate, is_leaky_relu, is_batch_norm, n_filters_start, is_single_text)
model_location = '/opt/program/ar-cnn-image/checkpoints/{}.hdf5'.format(model_full_name)
log_dir = '/opt/program/ar-cnn-image/logs/{}'.format(model_full_name)
print(log_dir)
print(model_location)

import os
import shutil
import time

class EvaluateCallback(keras.callbacks.Callback):
    def __init__(self, image_shape, sample_dir):
        self.image_shape = image_shape
        self.sample_dir = sample_dir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % epochs_per_sample == 0:
            sample_sqrt = 3
            generated_images = []
            for i in range(sample_sqrt**2):
                directory = "images_{}".format(i)
                os.makedirs(directory, exist_ok=True)
                input_image = self.generate_noise()

                img, _ = self.inference(model, input_image, directory, 1500)
                generated_images.append(img)
          
            final_im = Image.new('RGB', (image_shape[0] * sample_sqrt, image_shape[1] * sample_sqrt))

            y_offset = 0
            for i in range(sample_sqrt):
                x_offset = 0
                new_im = Image.new('RGB', (image_shape[0] * sample_sqrt, image_shape[1]))
                for j in range(sample_sqrt):
                    im = deepcopy(generated_images[(i * sample_sqrt) + j])
                    new_im.paste(im, (x_offset, 0))
                    x_offset += image_shape[0]
                final_im.paste(new_im, (0, y_offset))
                y_offset += image_shape[0]
                
            os.makedirs(self.sample_dir, exist_ok=True)
            final_im.save(os.path.join(self.sample_dir, 'samples_epoch_{}.png'.format(epoch)))


    def generate_noise(self):
        input_image = np.full(self.image_shape, 0)
        input_image = input_image.astype(np.uint8)
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def inference(self, model, input_image, directory, iterations):        
        working_image = deepcopy(input_image)

        for i in range(iterations):
            softmax_predictions, sigmoid_predictions = model.predict(working_image)
            softmax_predictions = softmax_predictions.flatten()
            sigmoid_predictions = sigmoid_predictions.reshape(-1, 256)

            softmax_predictions = softmax_predictions - np.max(softmax_predictions)
            softmax_predictions = np.exp(softmax_predictions)
            softmax_predictions = softmax_predictions / np.sum(softmax_predictions)
            indices = np.arange(softmax_predictions.shape[0])

            index = np.random.choice(indices, p=softmax_predictions)
            working_image = working_image.flatten()

            sigmoid_probs = sigmoid_predictions[index]
            sigmoid_probs = sigmoid_probs - np.max(sigmoid_probs)
            sigmoid_probs = np.exp(sigmoid_probs)
            sigmoid_probs = sigmoid_probs / np.sum(sigmoid_probs)
            sigmoid_indices = np.arange(sigmoid_probs.shape[0])
            working_image[index] = np.random.choice(sigmoid_indices, p=sigmoid_probs)

            working_image = np.reshape(working_image, [1, *self.image_shape])

        final_image = working_image
        img = create_image(final_image, os.path.join(directory, 'final.png'), image_shape=self.image_shape[:-1])
        return img, deepcopy(final_image)

import os

model = unet_model(input_size=image_shape, n_filters_start=n_filters_start, is_grayscale=is_grayscale, num_sub_layers=num_sub_layers, learning_rate=learning_rate)

resume_training = False
if resume_training:
    model = keras.models.load_model(model_location)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=model_location,
    monitor='val_loss',
    save_weights_only=False,
    verbose=1,
    mode='min',
    save_best_only=False)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')
evaluate_callback = EvaluateCallback(image_shape, log_dir)

if True:
    if is_single:
        history = model.fit(
            training_generator,
            # validation_data=validation_data,
            verbose=1,
            shuffle=True,
            steps_per_epoch=steps_per_epoch,
            epochs=15000,
            callbacks=[model_checkpoint_callback, evaluate_callback, tensorboard_callback])#, tensorboard_callback])
    else:
        history = model.fit(
            training_generator,
            validation_data=validation_data,
            verbose=1,
            shuffle=True,
            steps_per_epoch=steps_per_epoch,
            epochs=10000,
            callbacks=[model_checkpoint_callback, tensorboard_callback, evaluate_callback])#, tensorboard_callback])
    #epochs=cfg.epochs,
    #callbacks=callbacks)
# model.save("sc-model.hdf5")

"""Current Experiment: Full dataset NADE only 11% accuracy at this point (~30 epochs). ~17% accuracy at 60 epochs.

Experiments:
- at ~20k epochs, upper-bound-1 removal worked for 10 samples
- Tried out 1000 samples and seeing if that gives good results. upper-bound-1. Ok. It seems the numerical instability causes the loss to spike at random locations throughout the graph. This leads to poor results.
- Trying out 10 samples with 2.5 % removal. Also from now on switching to numerically stable softmax. nade-cifar-grayscale-double-softmax-num-samples-10-noise-upper-2.5-single.hdf5
  - This resulted in good results . ~85% accuracy at 17577 epochs. Image saved.

- Trying out 10 samples with 2.5 % removal with a larger model (~3 M parameters) with a deeper (2 sub layers) network. cifar-grayscale-double-softmax-num-samples-10-noise-upper-2.5-num-sub-layers-2-single.hdf5. Let's see at ~15000 epochs where the accuracy is.
"""
