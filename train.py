import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

from model import BaseModel
from utils import load_data, plot_accuracy, plot_loss

import os 
import argparse


def training(args):
    base_model = BaseModel(args.type_model, args.image_C, args.image_W, args.image_H, args.num_classes)
    loss_func = [keras.losses.SparseCategoricalCrossentropy(from_logits=True),]
    metrics = ['accuracy']
    model = base_model(loss_func, metrics)
    train = load_data(data_path = args.train_path)
    valid = load_data(data_path = args.valid_path)
    
    early_stopping = EarlyStopping(filepath = os.path.join(args.save_path, 'weight'),
                       save_weights_only=True,
                       monitor='val_accuracy',
                       mode='max',
                       save_best_only = True,
                       patience=5)
    
    history = model.fit(train,
                        epochs = args.num_epochs,
                        verbose = 2,
                        callbacks = [early_stopping],
                        validation_data = valid)

    return history 

def parse_args(parser):
    parser.add_argument('--train_path', required=True, help='path train dataset')
    parser.add_argument('--valid_path', required=True, help='path val dataset')
    parser.add_argument('--type_model', type=str, default='LeNet5')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--image_C', type=int, default=3)
    parser.add_argument('--image_H', type=int, default=400)
    parser.add_argument('--image_W', type=int, default=400)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--save_path',type=str, default='./save')
    
    args = parser.parse_args()
    return args 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    history = training(args)

    plot_loss(history, args.save_path)
    plot_accuracy(history, args.save_path)