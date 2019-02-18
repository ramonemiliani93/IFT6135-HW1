import argparse

import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

import os
import sys
sys.path.append('..')

from utils.classes import Params
from processing.loader import fetch_dataloader


def eval_model(dataset_dir, model_path):

    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.

    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load your best model
    print("\nLoading model from", model_path)
    model = torch.load(model_path, map_location=device)

    # Get params configuration file for data loader
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments', 'base_model', 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Extract test data loader
    dataloaders = fetch_dataloader('test', params, dataset_dir)
    test_dataloader = dataloaders.get('test')

    # Labels
    labels = {
        0: 'Cat',
        1: 'Dog'
    }

    # Predictions
    y_pred = []

    for data_batch, labels_batch in test_dataloader:
        data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output and loss
        output_batch = model(data_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()

        output_batch = 1 / (1 + np.exp(-output_batch))
        predictions = np.round(output_batch)
        y_pred.append(list(map(lambda x: labels[x], predictions)))

    return np.concatenate(y_pred, axis=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--results_dir", type=str, default='')

    args = parser.parse_args()
    image_dir = args.image_dir
    model_path = args.model_path
    results_dir = args.results_dir

    print("\nEvaluating results ... ")
    y_pred = eval_model(image_dir, model_path)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = os.path.join(results_dir, 'eval_pred.csv')

    print('\nSaving results')
    df = pd.DataFrame.from_dict({'id': list(range(1, len(y_pred)+1)), 'label': y_pred})
    df.to_csv(results_fname, encoding='utf-8', index=False)
