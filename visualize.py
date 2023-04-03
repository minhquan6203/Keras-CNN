import tensorflow as tf
import matplotlib.pyplot as plt

class Visualize(object):
    def __init__(self, history, path_save):
        self.history = history
        self.path_save = path_save

    def plot_accuracy(self):
        plt.clf()
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'{self.path_save}/accuracy.jpg')


    def plot_loss(self):
        plt.clf()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'{self.path_save}/loss.jpg')
    
    def __call__(self):
        self.plot_accuracy()
        self.plot_loss()
