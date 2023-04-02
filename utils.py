import tensorflow as tf
import matplotlib.pyplot as plt


def load_data(data_path: str, image_W: int = 400, image_H: int = 400, batch_size: int = 128):
    
    dataloader = tf.keras.preprocessing.image_dataset_from_directory(
                    data_path,
                    labels="inferred",
                    label_mode="int", 
                    color_mode="rgb",
                    batch_size=batch_size,
                    image_size=(image_H, image_W),
                    shuffle=True,
                    seed=1234)
                    
    return dataloader


def plot_accuracy(history, path_save):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{path_save}/accuracy.jpg')


def plot_loss(history, path_save):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{path_save}/loss.jpg')


