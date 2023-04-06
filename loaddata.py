import tensorflow as tf

class LoadData(object):
    def __init__(self,image_W: int = 224, image_H: int = 224, batch_size: int = 128):
        self.image_W=image_W
        self.image_H=image_H
        self.batch_size=batch_size

    def load_data(self,data_path: str):
        # Define data augmentation transformations
        data_augmentation = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
          tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
          tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
          tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        ])
        
        # Load images from directory
        dataloader = tf.keras.preprocessing.image_dataset_from_directory(
                        data_path,
                        labels="inferred",
                        label_mode="int", 
                        color_mode="rgb",
                        batch_size=self.batch_size,
                        image_size=(self.image_H, self.image_W),
                        shuffle=True,
                        seed=1234)
        
        # Apply data augmentation
        dataloader = dataloader.map(lambda x, y: (data_augmentation(x), y))
                        
        return dataloader
        
    def __call__(self, data_path: str):
        load_data = self.load_data(data_path)
        return load_data
