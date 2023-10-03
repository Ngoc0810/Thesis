import numpy as np # linear algebra
import cv2
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU,Concatenate,Resizing
from tensorflow.keras.layers import Conv2DTranspose as Deconvolution
from tensorflow.keras.models import Model

from utils import read_image
from consts import *

class KaggleModel():
    # link to model
    def __init__(self, input):
        self.instantiate_model(input)
        self.model_colourization = Model(inputs = input, outputs = self.model)
        
        self.LEARNING_RATE = 0.001
        self.model_colourization.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE),
                                    loss='mean_squared_error')
        self.model_colourization.summary()
        
        self.CHECKPOINT_PATH = f"output/{self.name()}/cp-{{epoch:06d}}.ckpt"
                
        self.EPOCH_COUNT = 555
        self.STEP_PER_EPOCH = 38
        self.EPOCH_PER_CHECKPOINT = 5
        self.BATCH_SIZE = 500
        
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.CHECKPOINT_PATH, 
            verbose = 1, 
            save_weights_only = True,
            save_freq = self.EPOCH_PER_CHECKPOINT * self.STEP_PER_EPOCH)
        self.csv_logger = tf.keras.callbacks.CSVLogger(f'training_log/{self.name()}.log')
        
        self.checkpoint = tf.train.Checkpoint(self.model_colourization)
        latest_checkpoint = tf.train.latest_checkpoint("output")
        self.initial_epoch = 0
        if latest_checkpoint != None:
            self.checkpoint.restore(latest_checkpoint)
            self.initial_epoch = int(latest_checkpoint[-11:-5])

    def name(self):
        return 'kaggle_model'
        
    # return l channel and ab channels
    def load_datapoint(self, file):
        image, grey = read_image(file)
        if image is None:
            return None, None
        grey = grey.astype(np.float32)
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float32)
        l_channel = image_lab[:, :, 0] # l channel
        ab_channels = image_lab[:, :, 1:] #  ab channels
        ab_channels = (ab_channels - 128)/128 # scale from [0, 256] to [-1, 1]
        return l_channel, ab_channels

    
    
    def instantiate_model(self, input):
        self.model = Conv2D(16,(3,3),padding='same',strides=1)(input)
        self.model = LeakyReLU()(self.model)
        #self.model = Conv2D(64,(3,3), activation='relu',strides=1)(self.model)
        self.model = Conv2D(32,(3,3),padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = MaxPooling2D(pool_size=(2,2),padding='same')(self.model)
        
        self.model = Conv2D(64,(3,3),padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = MaxPooling2D(pool_size=(2,2),padding='same')(self.model)
        
        self.model = Conv2D(128,(3,3),padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        self.model = BatchNormalization()(self.model)
        
        self.model = Conv2D(256,(3,3),padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        self.model = BatchNormalization()(self.model)
        
        self.model = UpSampling2D((2, 2))(self.model)
        self.model = Conv2D(128,(3,3),padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        self.model = BatchNormalization()(self.model)
        
        self.model = UpSampling2D((2, 2))(self.model)
        self.model = Conv2D(64,(3,3), padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        #self.model = BatchNormalization()(self.model)
        
        # concat_ = Concatenate([self.model, in_]) 
        self.model = Concatenate()([self.model, input])
        
        self.model = Conv2D(64,(3,3), padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        self.model = BatchNormalization()(self.model)
        
        self.model = Conv2D(32,(3,3),padding='same',strides=1)(self.model)
        self.model = LeakyReLU()(self.model)
        #self.model = BatchNormalization()(self.model)
        
        self.model = Conv2D(2,(3,3), activation='tanh',padding='same',strides=1)(self.model)

    def fit(self, generator, validation = None):
        K.clear_session()
        def generator_wrapper(generator):
            for x_input, y_input, _ in generator:        
                x_input = x_i.reshape(1, HEIGHT, WIDTH, 1)
                y_input = y_i.reshape(1, HEIGHT, WIDTH, 2)        
                yield (x_input, y_input)
            
        self.model_colourization.fit(
            generator_wrapper(generator),
            epochs = self.EPOCH_COUNT,
            verbose = 1,
            steps_per_epoch = self.STEP_PER_EPOCH,
            batch_size = self.BATCH_SIZE,
            shuffle = True,
            callbacks = [self.cp_callback, self.csv_logger],
            validation_data = generator_wrapper(validation),
            validation_steps = self.STEP_PER_EPOCH,
            validation_freq = 1,
            initial_epoch = self.initial_epoch,
        )

    def get_output(self, file, image):
        l_channel, ab_channels = self.load_datapoint(file)
        predicted_ab = self.model_colourization.predict(l_channel.reshape(1, HEIGHT, WIDTH, 1))
        predicted_ab = predicted_ab * 128 + 128
        predicted_ab = predicted_ab.reshape(HEIGHT, WIDTH, 2)
    
        image_predict = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        image_predict[:,:,1:] = predicted_ab
        image_predict = cv2.cvtColor(image_predict, cv2.COLOR_Lab2RGB)
        return image_predict
    