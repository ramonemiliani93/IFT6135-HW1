"""Main file to train and validate the model"""

import argparse
import logging
import os

import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
from tqdm import tqdm

from utils.classes import Params
import utils.functions as f
from model.architecture import models
from processing.loader import fetch_dataloader
from model.metrics import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=os.path.join('data', 'train', 'trainset'),
                    help="Directory containing the dataset")
parser.add_argument('--model', help="Model to be trained")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'last'


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on all batches in one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    optimizer : torch.optim
        Optimizer for the model parameters.
    loss_fn :
        Loss function to be optimized.
    dataloader : DataLoader
        Dataloader.
    metrics : dict
        Dictionary containing metrics to be calculated.
    params : object
        Hyperparameter object.
    """

    # set model to training mode
    model.train()

    # Reset all metrics
    for metric in metrics.values():
        metric.reset()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            train_batch, labels_batch = train_batch.to(params.device), labels_batch.to(params.device)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch.float())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # update the average loss
            metrics['loss'].update(loss.item())

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                for metric in metrics.values():
                    metric(output_batch, labels_batch, params)

            t.set_postfix(loss='{:05.3f}'.format(metrics['loss'].get_average()))
            t.update()

    # Summary of metrics in log
    metrics_string = "".join([str(metric) for metric in metrics.values()])
    logging.info("- Train metrics: \n" + metrics_string)


def validate(model, loss_fn, dataloader, metrics, params):
    """Train the model on all validation data.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    loss_fn :
        Loss function to be optimized.
    dataloader : DataLoader
        Dataloader.
    metrics : dict
        Dictionary containing metrics to be calculated.
    params : object
        Hyperparameter object.

    Returns
    -------
    metrics : dict
        Dictionary with metric results for validation split.
    """

    # set model to evaluation mode
    model.eval()

    # Reset all metrics
    for metric in metrics.values():
        metric.reset()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda and torch.cuda.is_available():
            data_batch, labels_batch = data_batch.to(params.device), labels_batch.to(params.device)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch.float())

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        for metric in metrics.values():
            metric(output_batch, labels_batch, params)

    # Summary of metrics in log
    metrics_string = "".join([str(metric) for metric in metrics.values()])
    logging.info("- Val metrics : \n" + metrics_string)
    return metrics


def train_and_validate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, params,
                       model_dir, restore_file=None):
    """Train the model and validate every epoch.

    Parameters
    ----------
        model : torch.nn.Module
            the neural network
        train_dataloader : DataLoader
            a torch.utils.data.DataLoader object that fetches training data
        val_dataloader : DataLoader
            a torch.utils.data.DataLoader object that fetches validation data
        optimizer: torch.optim
            optimizer for parameters of model
        loss_fn:
            a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics : dict
            a dictionary of functions that compute a metric using the output and labels of each batch
        params : Params
            hyperparameters
        model_dir : string
            directory containi,ng config, weights and log
        restore_file : string optional
            name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        f.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Scheduler for learning rate
        if epoch in scheduler.get('epochs'):
            state_dict = optimizer.state_dict()
            param_groups = state_dict.get('param_groups')
            new_lr = param_groups[0].get('lr') * scheduler.get('dampening')
            param_groups[0].update({'lr': new_lr})
            state_dict.update({'param_groups': param_groups})
            optimizer.load_state_dict(state_dict)
            logging.info("- Updating learning rate to {}".format(new_lr))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics.get('train'), params)

        # Evaluate for one epoch on validation set
        val_metrics = validate(model, loss_fn, val_dataloader, metrics.get('eval'), params)

        val_acc = val_metrics['accuracy'].get_accuracy()
        is_best = val_acc >= best_val_acc

        # Save weights
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}
        f.save_checkpoint(model, state, is_best=is_best, checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            f.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        f.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    if params.cuda and torch.cuda.is_available():
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.random_seed)
    if params.cuda:
        torch.cuda.manual_seed(params.random_seed)

    # Set the logger
    f.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Log params folder
    logging.info("Experiment: {}".format(json_path))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = fetch_dataloader('train', params, args.data_dir)
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    assert args.model in models, "Provided model not defined on available models dictionary"
    model = models.get(args.model)(params)
    model = model.to(params.device)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # Fetch loss function and metrics
    loss_fn = BCEWithLogitsLoss(torch.Tensor(params.weights).to(params.device) if hasattr(params, 'weights') else None)
    metrics = metrics

    # Scheduler
    dampening = params.dampening
    scheduler_epochs = params.scheduler_epochs
    scheduler = {
        'dampening': dampening,
        'epochs': scheduler_epochs
    }

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_validate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, params,
                       args.model_dir, args.restore_file)
