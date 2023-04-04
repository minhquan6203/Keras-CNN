import argparse
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, confusion_matrix
from model import CNN_Model
from loaddata import LoadData




def eval(args):
    
    model = tf.keras.models.load_model(os.path.join(os.path.join(args.checkpoint_path, 'best_model.h5')))
    load_data = LoadData(args.image_W, args.image_H, args.batch_size)
    test_data = load_data(data_path=args.test_path)

    # Evaluate model on test data
    eval_results = model.evaluate(test_data)
    accuracy = eval_results[-1]

    # Calculate f1 score
    y_true = test_data.labels
    y_pred = model.predict(test_data).argmax(axis=-1)
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


