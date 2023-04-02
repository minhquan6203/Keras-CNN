import argparse

import tensorflow as tf
from tensorflow import keras

from model import CNN_Model
from utils import load_data


def eval(args):

    base_model = CNN_Model()

    loss_function = [keras.losses.SparseCategoricalCrossentropy(from_logits=True),]
    metrics = ['accuracy']

    model = base_model(loss_function, metrics)
    model.load_weights(args.checkpoint_path)

    test = load_data(data_path = args.test_path)

    res_eval =  model.evaluate(test)
    score = res_eval[-1]
    return score


def parse_args(parser):
    parser.add_argument('--test_path', required=True, help='path test dataset')
    parser.add_argument('--checkpoint_path', required=True, help='path checkpoint')
    
    args = parser.parse_args()
    return args 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    score = eval(args)
    print("Test Accuracy: {}".format(score))