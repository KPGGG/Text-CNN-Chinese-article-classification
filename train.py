import tensorflow as tf
import json
import os
import argparse

from data_process import text_cnn_data_process
from text_cnn_model import TextCNN

split_rate = 0.2

class TextCNNTrainer:
    def __init__(self, args):
        self.args = args
        with open(self.args.config_path, 'r') as f:
            self.config = json.load(f)

        self.train_data_obj = self.load_data()
        #self.eval_data_obj = self.load_data(is_train=False)

        self.all_data, self.all_labels = self.train_data_obj.gen_data()

        self.train_data = self.all_data[:int(len(self.all_data)*split_rate)]
        self.train_labels = self.all_labels[:int(len(self.all_data)*split_rate)]

        self.eval_data = self.all_data[int(len(self.all_data)*split_rate):]
        self.eval_labels = self.all_labels[int(len(self.all_data)*split_rate):]


        self.model = self.create_model()

    def load_data(self, is_train = True):
        data_loader_obj = text_cnn_data_process(self.config["text_path"],
                                                self.config["stop_word_path"],
                                                self.config["output_path"],
                                                self.config["low_freq"],
                                                is_training= is_train)
        return data_loader_obj

    def create_model(self):
        model = TextCNN(config=self.config,
                        vocab_size=self.train_data_obj.vocab_size)
        return model

    def train(self):
