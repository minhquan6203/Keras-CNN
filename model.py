from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Flatten, Dense
from keras.regularizers import l2

class BaseModel(object):

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
                #layer1
                layers.Conv2D(filters=96, kernel_size=(11,11),strides=4, padding='valid', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),
                layers.Normalization(),
                #layer2
                layers.Conv2D(filters=256, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),
                layers.Normalization(),
                #layer3
                layers.Conv2D(filters=384, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                #layer4
                layers.Conv2D(filters=384, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                #layer5
                layers.Conv2D(filters=256, kernel_size=(3,3),strides=1, padding='same', activation='relu'),
                layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same'),        
                layers.Flatten(),
                #fc layer1
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                #fc layer2
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                #output layer
                layers.Dense(10, activation='softmax'),
            ])

        return model

    def VGG16(self):
        model = keras.Sequential(
            [   layers.Input((self.image_W, self.image_H, self.image_C)),
                #layer 1
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.Conv2D( 64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 2
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 3
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 4
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                # layer 5
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2), strides=(2, 2)),

                layers.Flatten(),
                # fc layer 1
                layers.Dense(4096, activation='relu'),
                #fc layer 2
                layers.Dense(4096, activation='relu'),
                #output layer
                layers.Dense(self.num_classes, activation='softmax'),
            ])

        return model


    def conv_block(self,x, filters, kernel_size, strides):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
        x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same', activation='relu')(x)
        x = Add()([x, shortcut])
        return x

    def identity_block(self,x, filters, kernel_size):
        x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
        x = Add()([x, x])
        return x

    def ResNet34(self):
        input_tensor = layers.Input((self.image_W, self.image_H, self.image_C)),
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



    def _define_model(self):
        
        if self.type_model=="AlexNet":
            return self.AlexNet()
        if self.type_model=="VGG16":
            return self.VGG16()
        if self.type_model=="ResNet34":
            return self.ResNet34()

        return self.LeNet5()


    
    def __call__(self, loss, metrics):
        model = self._define_model()

        model.compile(
            optimizer = keras.optimizers.Adam(),
            loss = loss,
            metrics = metrics,
        )

        return model
