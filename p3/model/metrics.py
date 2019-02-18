import numpy as np


class Metric(object):
    """Base class for metric implementation"""
    def reset(self):
        """Resets all attributes of the class to 0"""
        for var in vars(self):
            setattr(self, var, 0)

    def dump(self):
        """Dump class attributes on a dictionary for JSON compatibility when saving

        Returns
        -------
        dict : dict
            Dictionary with attributes as keys and information as value.
        """
        return {var: str(getattr(self, var)) for var in vars(self)}


class LossAverage(Metric):
    """Running average over loss value"""
    def __init__(self):
        super()
        self.steps = None
        self.total = None

    def __call__(self, output, labels, params):
        pass

    def __str__(self):
        """Return average loss when printing the class

        Returns
        -------
        str : str
            String containing the average loss for the current epoch.
        """
        return "Loss: {:0.3f}\n".format(self.total / self.steps)

    def update(self, val):
        """Update the loss average with the latest value

        Parameters
        ----------
        val : float
            Latest value of the loss after evaluating the models results.
        """
        self.steps += 1
        self.total += val

    def get_average(self):
        return self.total / self.steps


class Accuracy(Metric):
    """Computes the accuracy for a given set of predictions"""
    def __init__(self):
        super()
        self.correct = None
        self.total = None

    def __call__(self, outputs, labels, params):
        """Updates the number of correct and total samples given the outputs of the model and the correct labels.

        Note:
        Adding the softmax function is not necessary to calculate the accuracy.

        Parameters
        ----------
        outputs : ndarray
            Model predictions.
        labels : ndarray
            Correct outputs.
        params : object
            Parameter class with general experiment information.
        """
        outputs = 1 / (1 + np.exp(-outputs))
        outputs = np.round(outputs)
        self.total += float(labels.size)
        self.correct += np.sum(outputs == labels)

    def __str__(self):
        """Return sample accuracy when printing the class

        Returns
        -------
        str : str
            String containing the sample average accuracy for the summary steps.
        """
        return "Sample accuracy: {:0.3f} -- ({:.0f}/{:.0f})\n".format(self.correct/self.total, self.correct, self.total)

    def get_accuracy(self):
        """Returns average accuracy.

        Returns
        -------
        float : float
            Current accuracy for the sampled data.
        """
        return self.correct / self.total


# Maintain all metrics required in this dictionary - these are used in the training and evaluation loops
# key accuracy must be kept to select best model in evaluation
metrics = {
    'train': {
        'accuracy': Accuracy(),
        'loss': LossAverage()
    },
    'eval': {
        'accuracy': Accuracy(),
    }
}
