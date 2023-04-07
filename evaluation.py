import argparse
import os
import tensorflow as tf
from tensorflow import keras
from model import CNN_Model
from loaddata import LoadData

from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

def load_data(data_path: str, img_W: int = 224, img_H: int = 224, batch_size: int = 128):
      dataloder = tf.keras.preprocessing.image_dataset_from_directory(
                      data_path,
                      labels="inferred",
                      label_mode="int", 
                      color_mode="rgb",
                      batch_size=batch_size,
                      image_size=(img_H, img_W),
                      shuffle=True,
                      seed=1111)
                      
      return dataloder

def eval(args):

    model = tf.keras.models.load_model(os.path.join(os.path.join(args.checkpoint_path, 'best_model.h5')))

    test_data = load_data(args.test_path,args.image_W, args.image_H, args.batch_size)

    # Evaluate model on test data
    eval_results = model.evaluate(test_data)
    accuracy = eval_results[-1]

    # Calculate f1 score
    y_true = []
    y_pred = []
    for images, labels in test_data:
        y_true.extend(labels.numpy())
        y_pred.extend(model.predict(images).argmax(axis=-1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, f1, cm




def parse_args(parser):
    parser.add_argument('--test_path', required=True, help='path to test dataset')
    parser.add_argument('--image_H', type=int, default=224, help='height of input images')
    parser.add_argument('--image_W', type=int, default=224, help='width of input images')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for test data generator')
    parser.add_argument('--checkpoint_path', required=True, help='path to model checkpoint file')
    
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

# Evaluate model on test data and print results
    accuracy, f1, cm = eval(args)
    print("Test accuracy: {}".format(accuracy))
    print("Test f1 score: {}".format(f1))
    print("Confusion matrix:")
    print(cm)


