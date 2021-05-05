import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import RNNclassifier, CNNclassifier
from trainer import Trainer
from data_loader import DataLoader


def define_argparse():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--cnn', action='store_true')
    p.add_argument('--file_path', type=str, required=True)
    p.add_argument('--file_fmt', type=str, default='tsv')

    p.add_argument('--gpu_id', type=int, default= 0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--emb_dim', type=int, default=32)
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=256)
    # rnn
    p.add_argument('--hidden_size', type=int, default=32)
    p.add_argument('--n_layers', type=int, default=3)
    # cnn
    p.add_argument('--window_sizes', type=list, default=[2, 3, 4])
    p.add_argument('--n_filters', type=int, default=32)
    p.add_argument('--use_padding', type=bool, default=True)
    p.add_argument('--use_batchnorm', type=bool, default=True)

    config = p.parse_args()

    return config

def run_trainer(model, train_loader, valid_loader):
    print('=' * 35, 'Model description', '=' * 35)
    print(model, '\n')

    loss = nn.NLLLoss()  # expect 1d tensor
    optimizer = optim.Adam(model.parameters())

    print('=' * 37, 'Start training', '=' * 37)
    trainer = Trainer(config)
    trainer.train(model, optimizer, loss, train_loader, valid_loader)

def main(config) :
    dataloader = DataLoader()
    train_loader, valid_loader = dataloader.get_loaders(config)

    print('=' * 37, 'Dataset size', '=' * 38)
    print('Train size : {} Valid size : {}'.format(
        len(train_loader.dataset),
        len(valid_loader.dataset),
    ), '\n')

    input_size = len(dataloader.text.vocab)
    n_classes = len(dataloader.label.vocab)


    if config.rnn + config.cnn == 0:
        raise NotImplementedError('Model should be defined')

    if config.rnn:
        rnn_model = RNNclassifier(input_size, config.emb_dim, config.hidden_size,
                              config.n_layers, n_classes, config.dropout)
        run_trainer(rnn_model, train_loader, valid_loader)

    if config.cnn:
        cnn_model = CNNclassifier(input_size, config.emb_dim, config.window_sizes,
                              config.n_filters, config.use_batchnorm,
                              config.dropout, n_classes)
        run_trainer(cnn_model, train_loader, valid_loader)

    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None,
        'cnn': cnn_model.state_dict() if config.cnn else None,
        'config': config,
        'vocab': dataloader.text.vocab,
        'classes': dataloader.label.vocab,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparse()
    main(config)
