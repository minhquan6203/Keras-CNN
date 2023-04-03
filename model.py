from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import concatenate
from keras.regularizers import l2

class CNN_Model(object):

    def __init__(self, type_model: str ="Lenet5", image_C: int = 3, image_W: int = 400, image_H: int = 400, num_classes: int = 12):
        self.type_model=type_model
        self.image_C = image_C
        self.image_W = image_W
        self.image_H = image_H
        self.num_classes = num_classes
        
    def LeNet5(self):
        
        model = keras.Sequential(
            [   
                layers.Input((self.image_W, self.image_H, self.image_C)),
                layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='relu'),
                layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
                layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='relu'),
                layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
                layers.Flatten(),
                layers.Dense(120, activation='relu'),
                layers.Dense(84, activation='relu'),
                layers.Dense(self.num_classes,activation='softmax'),
            ])

        return model
    
    def AlexNet(self):
        model = keras.Sequential(
            [   layers.Input((self.image_W, self.image_H, self.image_C)),
                #Stage 1
                layers.Conv2D(filters=96, kernel_size=(11,11),strides=4, padding='valid', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),
                layers.Normalization(),
                #Stage 2
                layers.Conv2D(filters=256, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),
                layers.Normalization(),
                #Stage 3
                layers.Conv2D(filters=384, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                #Stage 4
                layers.Conv2D(filters=384, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                #Stage 5
                layers.Conv2D(filters=256, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),        
                layers.Flatten(),
                #fc layer1
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                #fc layer2
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.3),
                #output layer
                layers.Dense(self.num_classes, activation='softmax'),
            ])

        return model

    def VGG16(self):
        model = keras.Sequential(
            [   
                # Stage 1
                keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(self.image_W, self.image_H, self.image_C)),
                keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # Stage 2
                keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # Stage 3
                keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # Stage 4
                keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # Stage 5
                keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

                keras.layers.Flatten(),
                # fc layer 1
                keras.layers.Dense(4096, activation='relu'),
                keras.layers.Dropout(0.5),
                # fc layer 2
                keras.layers.Dense(4096, activation='relu'),
                keras.layers.Dropout(0.3),
                # output layer
                keras.layers.Dense(self.num_classes, activation='softmax'),
            ])

        return model



    def conv_block(self, x, filters, kernel_size, strides):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
        x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
        shortcut = Conv2D(filters, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = Add()([x, shortcut])
        return x

    def identity_block(self, x, filters, kernel_size):
        x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
        x = Add()([x, x])
        return x

    def ResNet34(self):
        input_tensor = Input(shape=(self.image_W, self.image_H, self.image_C))
        # Stage 1
        x = Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        # Stage 2
        x = self.conv_block(x, filters=64, kernel_size=3, strides=1)
        x = self.identity_block(x, filters=64, kernel_size=3)
        x = self.identity_block(x, filters=64, kernel_size=3)

        # Stage 3
        x = self.conv_block(x, filters=128, kernel_size=3, strides=2)
        x = self.identity_block(x, filters=128, kernel_size=3)
        x = self.identity_block(x, filters=128, kernel_size=3)
        x = self.identity_block(x, filters=128, kernel_size=3)

        # Stage 4
        x = self.conv_block(x, filters=256, kernel_size=3, strides=2)
        x = self.identity_block(x, filters=256, kernel_size=3)
        x = self.identity_block(x, filters=256, kernel_size=3)
        x = self.identity_block(x, filters=256, kernel_size=3)
        x = self.identity_block(x, filters=256, kernel_size=3)
        x = self.identity_block(x, filters=256, kernel_size=3)
        
        # Stage 5
        x = self.conv_block(x, filters=512, kernel_size=3, strides=2)
        x = self.identity_block(x, filters=512, kernel_size=3)
        x = self.identity_block(x, filters=512, kernel_size=3)
        
        # Output
        x = Flatten()(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(input_tensor, output_tensor)
        return model



    def dense_block(self,x, blocks, growth_rate):
        for i in range(blocks):
            conv = Conv2D(4 * growth_rate, (1,1), padding='same', kernel_initializer='he_normal')(x)
            conv = BatchNormalization(axis=3)(conv)
            conv = Activation('relu')(conv)

            conv = Conv2D(growth_rate, (3,3), padding='same', kernel_initializer='he_normal')(conv)
            conv = BatchNormalization(axis=3)(conv)
            conv = Activation('relu')(conv)

            x = concatenate([x, conv], axis=3)

        return x

    def transition_layer(self,x, compression):
        num_filters = int(x.shape.as_list()[-1] * compression)

        x = Conv2D(num_filters, (1,1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((2,2), strides=(2,2))(x)

        return x

    def DenseNet(self, blocks=[6,12,24,16], growth_rate=32, compression=0.5):


        inputs = Input(shape=(self.image_W, self.image_H, self.image_C))

        x = Conv2D(64, (7,7), strides=(2,2), padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

        for i, num_blocks in enumerate(blocks):
            x = self.dense_block(x, num_blocks, growth_rate)
            if i != len(blocks) - 1:
                x = self.transition_layer(x, compression)

        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((7,7))(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, x, name='DenseNet')

        return model


    def conv_block(inputs, num_filters, kernel_size=3, padding='same', strides=1, activation='relu'):
        conv = Conv2D(num_filters, kernel_size, padding=padding, strides=strides)(inputs)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
        conv = Conv2D(num_filters, kernel_size, padding=padding, strides=strides)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
        return conv

    def encoder_block(inputs, num_filters, pool_size=(2,2), dropout=0.2):
        conv = conv_block(inputs, num_filters)
        pool = MaxPooling2D(pool_size)(conv)
        if dropout > 0.0:
            pool = Dropout(dropout)(pool)
        return conv, pool

    def decoder_block(inputs, skip_features, num_filters, upsample_size=(2,2), dropout=0.2):
        upsample = UpSampling2D(upsample_size)(inputs)
        upconv = Conv2D(num_filters, 2, padding='same')(upsample)
        upconv = BatchNormalization()(upconv)
        skip_conn = concatenate([upconv, skip_features], axis=3)
        conv = conv_block(skip_conn, num_filters)
        if dropout > 0.0:
            conv = Dropout(dropout)(conv)
        return conv

    def build_unet(input_shape, num_classes, base_filters=64, depth=4, dropout=0.2):
        inputs = Input(input_shape)

        skip_connections = []
        conv = inputs
        for i in range(depth):
            conv, pool = encoder_block(conv, base_filters*(2**i), dropout=dropout)
            skip_connections.append(conv)

        conv = conv_block(pool, base_filters*(2**depth), dropout=dropout)

        for i in reversed(range(depth)):
            conv = decoder_block(conv, skip_connections[i], base_filters*(2**i), dropout=dropout)

        outputs = Conv2D(num_classes, 1, activation='softmax')(conv)

        model = Model(inputs=inputs, outputs=outputs, name='U-Net')
        return model



    def define_model(self):
        
        if self.type_model=="AlexNet":
            return self.AlexNet()
        if self.type_model=="VGG16":
            return self.VGG16()
        if self.type_model=="ResNet34":
            return self.ResNet34()
        if self.type_model=="DenseNet":
            return self.DenseNet()

        return self.LeNet5()


    
    def __call__(self, loss_function, metrics):
        model = self.define_model()

        model.compile(
            optimizer = keras.optimizers.Adam(),
            loss = loss_function,
            metrics = metrics,
        )

        return model
