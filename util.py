import os
import re
import json
import pickle
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

from torchtext import data as ttd
from torchtext.data import Example, Dataset
from torch.utils.data import DataLoader

import numpy as np
import utils

def convert_to_dataset(data, eng):
    """
    Pre-process input DataFrame and convert pandas DataFrame to torchtext Dataset.
    Args:
        data: (DataFrame) pandas DataFrame to be converted into torchtext Dataset
        kor: torchtext Field containing Korean sentence
        eng: torchtext Field containing English sentence

    Returns:
        (Dataset) torchtext Dataset containing 'kor' and 'eng' Fields
    """
    # drop missing values not containing str value from DataFrame
    missing_rows = [idx for idx, row in data.iterrows() if type(row.input) != str]
    data = data.drop(missing_rows)

    # convert each row of DataFrame to torchtext 'Example' containing 'kor' and 'eng' Fields
    list_of_examples = [Example.fromlist(row.apply(lambda x: clean_text(x)).tolist(),
                                         fields=[('input', eng), ('target', eng)]) for _, row in data.iterrows()]

    # construct torchtext 'Dataset' using torchtext 'Example' list
    dataset = Dataset(examples=list_of_examples, fields=[('input', eng), ('target', eng)])

    return dataset

def load_dataset(mode):
    """
    Load train, valid and test dataset as a pandas DataFrame
    Args:
        mode: (string) configuration mode used to which dataset to load

    Returns:
        (DataFrame) train, valid, test dataset converted to pandas DataFrame
    """
    if mode == 'train':
        with open('data/train_ex.json', encoding='utf-8') as f:
            examples = [json.loads(line) for line in f]

        train_dataset = utils.Dataset(examples)

        with open('data/val_ex.json', encoding='utf-8') as f:
            examples = [json.loads(line) for line in f]
        val_dataset = utils.Dataset(examples)

        embed = torch.Tensor(np.load('data/embedding.npz')['embedding'])
        with open('data/word2id.json') as f:
            word2id = json.load(f)
        vocab = utils.Vocab(embed, word2id)

        return vocab, train_dataset, val_dataset

    else:
        test_file = os.path.join('data/', 'test_long.csv')
        test_data = pd.read_csv(test_file, encoding='utf-8')

        print(f'Number of testing examples: {len(test_data)}')

        return test_data


def clean_text(text):
    """
    remove special characters from the input sentence to normalize it
    Args:
        text: (string) text string which may contain special character

    Returns:
        normalized sentence
    """
    text = re.sub('[-=+,#\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]`…》]', '', text)
    return text



def make_iter(batch_size, mode, train_data=None, valid_data=None, test_data=None):
    # convert pandas DataFrame to torchtext dataset
    if mode == 'train':
        #print(train_data[0])
        train_iter = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True)
        val_iter = DataLoader(dataset=valid_data,
                              batch_size=batch_size,
                              shuffle=False)


        return train_iter, val_iter

    else:
        test_data = convert_to_dataset(test_data, eng)

        # defines dummy list will be passed to the BucketIterator
        dummy = list()

        # make iterator using test dataset
        print(f'Make Iterators for testing . . .')
        test_iter, _ = ttd.BucketIterator.splits(
            (test_data, dummy),
            sort_key=lambda sent: len(sent.kor),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return test_iter


def epoch_time(start_time, end_time):
    """
    Calculate the time spent to train one epoch
    Args:
        start_time: (float) training start time
        end_time: (float) training end time

    Returns:
        (int) elapsed_mins and elapsed_sec spent for one epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def display_attention(candidate, translation, attention):
    """
    displays the model's attention over the source sentence for each target token generated.
    Args:
        candidate: (list) tokenized source tokens
        translation: (list) predicted target translation tokens
        attention: a tensor containing attentions scores
    Returns:
    """
    # attention = [target length, source length]

    attention = attention.cpu().detach().numpy()
    # attention = [target length, source length]

    font_location = 'pickles/NanumSquareR.ttf'
    fontprop = fm.FontProperties(fname=font_location)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + [t.lower() for t in candidate], rotation=45, fontproperties=fontprop)
    ax.set_yticklabels([''] + translation, fontproperties=fontprop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


class Params:
    """
    Class that loads hyperparameters from a json file
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)
        self.load_vocab()

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def load_vocab(self):
        # load kor and eng vocabs to add vocab size configuration
        embed = torch.Tensor(np.load('data/embedding.npz')['embedding'])

        with open('data/word2id.json') as f:
            word2id = json.load(f)
        vocab = utils.Vocab(embed, word2id)

        # add device information to the the params
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # add <sos> and <eos> tokens' indices used to predict the target sentence
        params = {'input_dim': embed.size(0), 'output_dim': embed.size(0),
                  'sos_idx': vocab.w2i('<sos>'), 'eos_idx': vocab.w2i('<eos>'),
                  'pad_idx': vocab.w2i('<pad>'), 'device': device}

        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
