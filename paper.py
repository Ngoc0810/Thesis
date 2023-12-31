import numpy as np # linear algebra
import cv2
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU,Concatenate,Resizing
from tensorflow.keras.layers import Conv2DTranspose as Deconvolution
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt

from utils import read_image
from consts import *

class PaperModel():
    # https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/twostage-pyramidal-convolutional-neural-networks-for-image-colorization/07436C294A15C1CB9A694E42CB51C99D
    def modified_MAE(y_true, y_pred):
        # h_loss = abs(h_true - h_pred) if abs(h_true - h_pred) <= 1
        # h_loss = 2 - abs(h_true - h_pred)  if abs(h_true - h_pred) > 1
        # -> h_loss = min (abs(h_true - h_pred), 2 - abs(h_true - h_pred))
        # s_loss = abs(s_true - s_pred)
        diff = math_ops.abs(math_ops.subtract(y_true, y_pred))
        s_channel = diff[:, :, :, 1]
        h_channel = diff[:, :, :, 0]
        h_inverted = math_ops.subtract(tf.constant([2.0]), h_channel)
        h_correct = math_ops.minimum(h_channel, h_inverted)
        return h_correct + s_channel
    
    def __init__(self, image):
        self.data_point_cache = dict()
        self.LEARNING_RATE = 0.001
        self.CHECKPOINT_PATH_LSRN = f"output/{self.name()}/lsrn/cp-{{epoch:06d}}.ckpt"        
        self.CHECKPOINT_PATH_COLOR = f"output/{self.name()}/color/cp-{{epoch:06d}}.ckpt"        
        # if epoch < LSRN_EPOCH_COUNT then trains only LSRN
        self.LSRN_EPOCH_COUNT = 1
        # otherwise train and finetune
        self.COLOR_EPOCH_COUNT = 1
        self.MODE = 'lsrn'
        assert self.MODE == 'lsrn' or self.MODE == 'color'

        # STEP_PER_EPOCH should be set to the amount of data point total
        self.STEP_PER_EPOCH = 100
        # VALIDATION_STEP_PER_EPOCH should be set to the amount of data point in the validation set
        self.VALIDATION_STEP_PER_EPOCH = 20
        # BATCH_SIZE should be set to a divisor of STEP_PER_EPOCH, although it's not too important if batch size is small enough
        self.BATCH_SIZE = 25
        
        self.EPOCH_PER_CHECKPOINT = 20
        self.instantiate_model(image)
        self.csv_logger_lsrn = tf.keras.callbacks.CSVLogger(f'training_log/{self.name()}_lsrn.log', append=True)
        self.csv_logger_color = tf.keras.callbacks.CSVLogger(f'training_log/{self.name()}_color.log', append=True)
        self.cp_callback_lsrn = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.CHECKPOINT_PATH_LSRN, 
            verbose = 1, 
            save_weights_only = True,
            save_freq = self.EPOCH_PER_CHECKPOINT * self.STEP_PER_EPOCH)
        
        self.cp_callback_color = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.CHECKPOINT_PATH_COLOR, 
            verbose = 1, 
            save_weights_only = True,
            save_freq = self.EPOCH_PER_CHECKPOINT * self.STEP_PER_EPOCH)

        self.checkpoint_lsrn = tf.train.Checkpoint(self.model_LSRN)
        self.checkpoint_color = tf.train.Checkpoint(self.model_color)
        self.initial_epoch = 0
        self.load_checkpoints()

    def load_checkpoints(self):
        latest_lsrn = tf.train.latest_checkpoint(f"output/{self.name()}/lsrn")
        latest_color = tf.train.latest_checkpoint(f"output/{self.name()}/color")
        lsrn_epoch = 0
        color_epoch = 0
        if latest_lsrn is not None:
            self.checkpoint_lsrn.restore(latest_lsrn)
            lsrn_epoch = int(latest_lsrn[-11:-5])
        if latest_color is not None:
            self.checkpoint_color.restore(latest_color)
            color_epoch = int(latest_color[-11:-5])
        
        self.initial_epoch = color_epoch
        print(f'Loading checkpoint {lsrn_epoch} for lsrn, checkpoint {color_epoch} for color')
        if lsrn_epoch < color_epoch: # need to load the color model on top of lsrn
            print(f'Overwriting lsrn weights with color weights')
            self.model_color.save_weights(f'{self.name()}_temp.keras')
            self.model_LSRN.load_weights(f'{self.name()}_temp.keras', by_name=True)
        elif color_epoch < lsrn_epoch:
            print(f'Overwriting color weights with lsrn weights')
            self.model_LSRN.save_weights(f'{self.name()}_temp.keras')
            self.model_color.load_weights(f'{self.name()}_temp.keras', by_name=True)
            self.initial_epoch = lsrn_epoch
        elif color_epoch != 0:
            raise Exception("Error: equal values of lsrn and color epoch")

    def name(self):
        return 'paper'
        
    # return v channel and hs channels
    def load_datapoint(self, file):
        if self.data_point_cache is None:
            self.data_point_cache = dict()
        if file in self.data_point_cache:
            return self.data_point_cache[file]
        image, _ = read_image(file)
        if image is None:
            self.data_point_cache[file] = (None, None, None)
            return (None, None, None)
        def get_v_hs(rgb_image):
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
            v_channel = hsv[:, :, 2]
            v_channel = (v_channel - 127.5) / 127.5 # [0, 255] -> [-1, 1]
            h_channel = hsv[:, :, 0]
            h_channel = (h_channel - 89.5) / 89.5  # [0, 179] -> [-1, 1]
            s_channel = hsv[:, :, 1]
            s_channel = (s_channel - 127.5) / 127.5 # [0, 255] -> [-1, 1]
            return v_channel, np.stack([h_channel, s_channel], axis=2)

        v_channel, hs_channels = get_v_hs(image)
        image_scaled = cv2.resize(image, (HEIGHT//4, WIDTH//4))
        _, low_resolution_hs_channels = get_v_hs(image_scaled)
        self.data_point_cache[file] = (v_channel, hs_channels, low_resolution_hs_channels)
        return (v_channel, hs_channels, low_resolution_hs_channels)

    def instantiate_model(self, image):
        self.LSRN = Conv2D(64, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_0')(image)
        self.LSRN = Conv2D(64, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_1')(self.LSRN)
        self.LSRN = BatchNormalization(name = 'lsrn_bn_1')(self.LSRN)
        self.LSRN = Conv2D(128, (3, 3), padding='same', strides=2, activation='relu', name = 'lsrn_conv_2')(self.LSRN)
        self.LSRN = Conv2D(128, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_3')(self.LSRN)
        self.LSRN = BatchNormalization(name = 'lsrn_bn_3')(self.LSRN)
        self.LSRN = Conv2D(256, (3, 3), padding='same', strides=2, activation='relu', name = 'lsrn_conv_4')(self.LSRN)
        self.LSRN = Conv2D(256, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_5')(self.LSRN)
        self.LSRN = BatchNormalization(name = 'lsrn_bn_5')(self.LSRN)
        self.LSRN = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_6')(self.LSRN)
        self.LSRN = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_7')(self.LSRN)
        self.LSRN = Conv2D(512, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_8')(self.LSRN)
        self.LSRN = BatchNormalization(name = 'lsrn_bn_8')(self.LSRN)
        self.LSRN = Conv2D(256, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_9')(self.LSRN)
        self.LSRN = Conv2D(256, (3, 3), padding='same', strides=1, activation='relu', name = 'lsrn_conv_10')(self.LSRN)
        self.LSRN = BatchNormalization(name = 'lsrn_bn_10')(self.LSRN)
        self.LSRN = Conv2D(2, (3, 3), padding='same', strides=1, activation='tanh', name = 'lsrn_conv_11')(self.LSRN)
        
        self.model_LSRN = Model(name = 'lsrn', inputs = image, outputs = self.LSRN)
        self.model_LSRN.compile(optimizer = Adam(learning_rate=self.LEARNING_RATE),
                                    loss = PaperModel.modified_MAE, run_eagerly=True)
        self.model_LSRN.summary()
        
        image_quarter = Resizing(HEIGHT//4, WIDTH//4)(image)
        image_quarter = Concatenate()([image_quarter, self.LSRN])
        image_quarter = Conv2D(64,(3,3),padding='same',strides=1, activation='relu')(image_quarter)
        image_quarter = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(image_quarter)
        image_quarter = BatchNormalization()(image_quarter)
        image_quarter = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(image_quarter)
        image_quarter = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(image_quarter)
        image_quarter = BatchNormalization()(image_quarter)
        image_quarter = Deconvolution(128, (3,3),padding='same', strides=2)(image_quarter)
        
        image_half = Resizing(HEIGHT//2, WIDTH//2)(image)
        LSRN_resized = Resizing(HEIGHT//2, WIDTH//2)(self.LSRN)
        image_half = Concatenate()([image_half, LSRN_resized])
        image_half = Conv2D(64,(3,3),padding='same',strides=1, activation='relu')(image_half)
        image_half = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(image_half)
        image_half = BatchNormalization()(image_half)
        image_half = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(image_half)
    
        self.RCN = Concatenate()([image_quarter, image_half])
        self.RCN = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(self.RCN)
        self.RCN = Deconvolution(128, (3,3),padding='same', strides=2, activation='relu')(self.RCN)
        self.RCN = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(self.RCN)
        self.RCN = Conv2D(128,(3,3),padding='same',strides=1, activation='relu')(self.RCN)
        self.RCN = BatchNormalization()(self.RCN)
        self.RCN = Conv2D(2,(3,3),padding='same',strides=1, activation='tanh')(self.RCN)

        
        self.model_color = Model(name = 'paper_colourization', inputs = image, outputs = self.RCN)
        # keep lsrn for easy training, no finetune is done
        for l in self.model_color.layers:
            if l.name[:4] == 'lsrn':
                l.trainable = False
                
        for l in self.model_color.layers:
            print(l.name, l.trainable)
        self.model_color.compile(optimizer = Adam(learning_rate=self.LEARNING_RATE),
                                    loss = PaperModel.modified_MAE, run_eagerly=False)
        self.model_color.summary()
        for k,v in self.model_color._get_trainable_state().items():
            print(k, v)
        # print(self.model_color._get_trainable_state())
    
    
    def fit(self, generator, validation):
        K.clear_session()
        def LSRN_generator(generator):
            for file in generator:
                x_i, _, y_i = self.load_datapoint(file)
                if x_i is None:
                    continue
                x_input = x_i.reshape(1, HEIGHT, WIDTH, 1)
                y_input = y_i.reshape(1, HEIGHT//4, WIDTH//4, 2)
                yield (x_input, y_input)
                
        def COLOR_generator(generator):
            for file in generator:
                x_i, y_i, _ = self.load_datapoint(file)
                if x_i is None:
                    continue
                x_input = x_i.reshape(1, HEIGHT, WIDTH, 1)
                y_input = y_i.reshape(1, HEIGHT, WIDTH, 2)        
                yield (x_input, y_input)
                
        self.model_LSRN.fit(
            LSRN_generator(generator),
            epochs = self.initial_epoch + self.LSRN_EPOCH_COUNT,
            verbose = 1,
            steps_per_epoch = self.STEP_PER_EPOCH,
            batch_size = self.BATCH_SIZE,
            shuffle = False,
            callbacks = [self.cp_callback_lsrn, self.csv_logger_lsrn],
            validation_data = LSRN_generator(validation),
            validation_steps = self.VALIDATION_STEP_PER_EPOCH,
            validation_freq = 1,
            initial_epoch = self.initial_epoch,
        )
        self.initial_epoch += self.LSRN_EPOCH_COUNT
        # load the newly trained LSRN on top of colorization
        self.model_LSRN.save_weights(f'{self.name()}_temp.keras')
        self.model_color.load_weights(f'{self.name()}_temp.keras', by_name=True)
        
        self.model_color.fit(
            COLOR_generator(generator),
            epochs = self.initial_epoch + self.COLOR_EPOCH_COUNT,
            verbose = 1,
            steps_per_epoch = self.STEP_PER_EPOCH,
            batch_size = self.BATCH_SIZE,
            shuffle = False,
            callbacks = [self.cp_callback_color, self.csv_logger_color],
            validation_data = COLOR_generator(validation),
            validation_steps = self.VALIDATION_STEP_PER_EPOCH,
            validation_freq = 1,
            initial_epoch = self.initial_epoch,
        )

    def get_output(self, file, image):
        if self.MODE == 'lsrn':
            return self.get_output_lsrn(file, image)
        else:
            return self.get_output_color(file, image)
    
    def get_output_color(self, file, image):
        v_channel, _, _ = self.load_datapoint(file)
        v_channel = (v_channel * 127.5) + 127.5
        predicted_hs = self.model_color.predict(v_channel.reshape(1, HEIGHT, WIDTH, 1))
        predicted_hs = predicted_hs.reshape(HEIGHT, WIDTH, 2)
        predicted_h = predicted_hs[:,:,0]
        predicted_h = predicted_h * 89.5 + 89.5
        predicted_s = predicted_hs[:,:,1]
        predicted_s = (predicted_s * 127.5) + 127.5
        assert (predicted_h <= 179).all() and (predicted_h >= 0).all(), "wrong predicted h"
        assert (predicted_s <= 255).all() and (predicted_s >= 0).all(), "wrong predicted s"
        assert (v_channel <= 255).all() and (v_channel >= 0).all(), "wrong v"
        hsv_image = np.stack([predicted_h, predicted_s, v_channel], axis = 2).astype(np.ubyte)
        image_predict = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return image_predict

    def get_output_lsrn(self, file, image):
        v_channel, _, correct = self.load_datapoint(file)
        v_channel = v_channel * 127.5 + 127.5
        lr_predicted_hs = self.model_LSRN.predict(v_channel.reshape(1, HEIGHT, WIDTH, 1))
        lr_predicted_hs = lr_predicted_hs.reshape(HEIGHT//4, WIDTH//4, 2)
        lr_predicted_h = lr_predicted_hs[:,:,0]
        lr_predicted_h = lr_predicted_h * 89.5 + 89.5
        lr_predicted_s = lr_predicted_hs[:,:,1]
        lr_predicted_s = lr_predicted_s * 127.5 + 127.5
        predicted_h = cv2.resize(lr_predicted_h, (HEIGHT, WIDTH))
        predicted_s = cv2.resize(lr_predicted_s, (HEIGHT, WIDTH))
        # return np.stack([predicted_s, predicted_s, predicted_s], axis = 2).astype(np.ubyte)
        
        assert (predicted_h <= 179).all() and (predicted_h >= 0).all(), "wrong predicted h"
        assert (predicted_s <= 255).all() and (predicted_s >= 0).all(), "wrong predicted s"
        assert (v_channel <= 255).all() and (v_channel >= 0).all(), "wrong v"
        # plt.imshow(predicted_h)
        hsv_image = np.stack([predicted_h, predicted_s, v_channel], axis = 2).astype(np.ubyte)
        image_predict = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return image_predict