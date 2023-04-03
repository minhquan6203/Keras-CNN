import tensorflow as tf
import matplotlib.pyplot as plt

class LoadData(object):
    def __init__(self,image_W: int = 224, image_H: int = 224, batch_size: int = 128):
        self.image_W=image_W
        self.image_H=image_H
        self.batch_size=batch_size

    def load_data(self,data_path: str):
        dataloader = tf.keras.preprocessing.image_dataset_from_directory(
                        data_path,
                        labels="inferred",
                        label_mode="int", 
                        color_mode="rgb",
                        batch_size=self.batch_size,
                        image_size=(self.image_H, self.image_W),
                        shuffle=True,
                        seed=1234)
                        
        return dataloader
        
    def __call__(self, data_path: str):
        load_data = self.load_data(data_path)
        return load_data
