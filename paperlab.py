import numpy as np # linear algebra
import cv2
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,Concatenate,Resizing,LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose as Deconvolution
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt

from utils import read_image
from consts import *

class PaperLabModel():
    # https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/twostage-pyramidal-convolutional-neural-networks-for-image-colorization/07436C294A15C1CB9A694E42CB51C99D
    # instead of using hsv, we use lab for ease of computation
    
    def __init__(self, image):
        self.data_point_cache = dict()
        self.LEARNING_RATE = 0.0001
        self.CHECKPOINT_PATH_LSRN = f"output/{self.name()}/lsrn/cp-{{epoch:06d}}.ckpt"        
        self.CHECKPOINT_PATH_COLOR = f"output/{self.name()}/color/cp-{{epoch:06d}}.ckpt"        
        # if epoch < LSRN_EPOCH_COUNT then trains only LSRN
        self.LSRN_EPOCH_COUNT = 0
        # otherwise train and finetune
        self.COLOR_EPOCH_COUNT = 0
        self.MODE = 'lsrn'
        self.MODE = 'color'
        assert self.MODE == 'lsrn' or self.MODE == 'color'

        # STEP_PER_EPOCH should be set to the amount of data point total
        self.STEP_PER_EPOCH = 100
        # VALIDATION_STEP_PER_EPOCH should be set to the amount of data point in the validation set
        self.VALIDATION_STEP_PER_EPOCH = 100
        # BATCH_SIZE should be set to a divisor of STEP_PER_EPOCH, although it's not too important if batch size is small enough
        self.BATCH_SIZE = 100
        
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
        print("Load checkpoints: ", lsrn_epoch, color_epoch)
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
        return 'paper_lab'
        
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
        def get_l_ab(rgb_image):
            # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
            lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab).astype(np.float32)
            l_channel = lab[:, :, 0]
            ab_channels = lab[:, :, 1:] #  ab channels
            l_channel = l_channel / 255 # scale from [0, 255] to [0, 1]
            ab_channels = (ab_channels - 128) / 127 # scale from [1, 255] to [-1, 1]
            return l_channel, ab_channels

        l_channel, ab_channels = get_l_ab(image)
        image_scaled = cv2.resize(image, (HEIGHT//4, WIDTH//4))
        _, low_resolution_ab_channels = get_l_ab(image_scaled)
        self.data_point_cache[file] = (l_channel, ab_channels, low_resolution_ab_channels)
        return (l_channel, ab_channels, low_resolution_ab_channels)

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
                                    loss = 'MSE', run_eagerly=False)
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
        self.RCN = Conv2D(2,(3,3),padding='same',strides=1, activation='tanh')(self.RCN)##

        
        self.model_color = Model(name = 'paper_colourization', inputs = image, outputs = self.RCN)
        # keep lsrn for easy training, no finetune is done
        for l in self.model_color.layers:
            if l.name[:4] == 'lsrn':
                l.trainable = False
                
        for l in self.model_color.layers:
            print(l.name, l.trainable)
        self.model_color.compile(optimizer = Adam(learning_rate=self.LEARNING_RATE),
                                    loss = 'MSE', run_eagerly=False)
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
        l_channel, _, _ = self.load_datapoint(file)
        l_channel = l_channel * 255
        predicted_ab = self.model_color.predict(l_channel.reshape(1, HEIGHT, WIDTH, 1))
        predicted_ab = predicted_ab.reshape(HEIGHT, WIDTH, 2)
        predicted_ab = predicted_ab * 127 + 128
        assert (predicted_ab <= 255).all() and (predicted_ab >= 1).all(), "wrong predicted ab"
        assert (l_channel <= 255).all() and (l_channel >= 0).all(), "wrong l"
        lab_image = np.stack([l_channel, predicted_ab[:,:,0], predicted_ab[:,:,1]], axis = 2).astype(np.ubyte)
        image_predict = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)
        return image_predict

    def get_output_lsrn(self, file, image):
        l_channel, _, correct = self.load_datapoint(file)
        l_channel = l_channel * 255
        lr_predicted_ab = self.model_LSRN.predict(l_channel.reshape(1, HEIGHT, WIDTH, 1))
        # lr_predicted_ab = correct
        lr_predicted_ab = lr_predicted_ab.reshape(HEIGHT//4, WIDTH//4, 2)
        lr_predicted_ab = lr_predicted_ab * 127 + 128
        
        predicted_ab = cv2.resize(lr_predicted_ab, (HEIGHT, WIDTH))
        # return np.stack([predicted_s, predicted_s, predicted_s], axis = 2).astype(np.ubyte)
        
        assert (predicted_ab <= 255).all() and (predicted_ab >= 1).all(), "wrong predicted ab"
        assert (l_channel <= 255).all() and (l_channel >= 0).all(), "wrong l"
        # plt.imshow(predicted_h)
        lab_image = np.stack([l_channel, predicted_ab[:,:,0], predicted_ab[:,:,1]], axis = 2).astype(np.ubyte)
        image_predict = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)
        return image_predict