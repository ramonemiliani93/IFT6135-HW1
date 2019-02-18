import numpy as np
import sklearn

data = np.load('mnist.npy')

X_train = data[0][0]
y_train = data[0][1]
X_valid = data[1][0]
y_valid= data[1][1]
X_test = data[2][0]
y_test = data[2][1]


class NeuralNetwork(object):
    def __init__(self, hidden_dims=(1024, 2048), n_hidden=2, epochs=10, mode='train', datapath=None, model_path=None,
                 activation_type='relu', initialization_type='Glorot', batch_size=None, step=0.03,
                 training_data=X_train, training_labels=y_train, test_data=X_valid, test_labels=y_valid):
        self.hidden_dims = hidden_dims
        self.n_hidden = n_hidden
        self.mode = mode
        self.datapath = datapath
        self.model_path = model_path
        self.epochs = epochs
        self.activation_type = activation_type
        self.initialization_type = initialization_type
        self.batch_size = batch_size
        self.cache = {'loss': []}
        self.parameters = {}
        self.step = step
        self.training_data = training_data
        self.test_data = test_data
        self.training_labels = training_labels
        self.test_labels = test_labels
        self.grads = {}

    def initialize_weights(self, initialization_type='Normal'):
        d1 = self.hidden_dims[0]
        d2 = self.hidden_dims[1]
        m = 10
        if initialization_type == 'Normal':
            W1 = np.random.randn(d1, 784)
            W2 = np.random.randn(d2, d1)
            W3 = np.random.randn(m, d2)
        elif initialization_type == 'Glorot':
            d1_s = np.sqrt(6 / (784 + d1))
            d2_s = np.sqrt(6 / (d1 + d2))
            d3_s = np.sqrt(6 / (d2 + m))
            W1 = np.random.uniform(-d1_s, d1_s, size=(d1, 784))
            W2 = np.random.uniform(-d2_s, d2_s, size=(d2, d1))
            W3 = np.random.uniform(-d3_s, d3_s, size=(m, d2))
        else:
            W1 = np.zeros((d1, 784))
            W2 = np.zeros((d2, d1))
            W3 = np.zeros((m, d2))

        b1 = np.zeros((d1, 1))
        b2 = np.zeros((d2, 1))
        b3 = np.zeros((m, 1))
        self.parameters['W1'] = W1
        self.parameters['W2'] = W2
        self.parameters['W3'] = W3
        self.parameters['b1'] = b1
        self.parameters['b2'] = b2
        self.parameters['b3'] = b3

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        self.cache['input'] = x
        self.cache['a1'] = np.dot(self.parameters['W1'], x.T) + self.parameters['b1']
        self.cache['h1'] = self.activation(self.cache['a1'], activation_type=self.activation_type)
        self.cache['a2'] = np.dot(self.parameters['W2'], self.cache['h1']) + self.parameters['b2']
        self.cache['h2'] = self.activation(self.cache['a2'], activation_type=self.activation_type)
        self.cache['a3'] = np.dot(self.parameters['W3'], self.cache['h2']) + self.parameters['b3']
        self.cache['probabilities'] = self.softmax(self.cache['a3'])
        if self.mode == 'test':
            print('probabilities: ', np.sum(self.cache['probabilities'], axis=0))

    @staticmethod
    def activation(x, activation_type='relu'):
        if activation_type.lower() == 'sigmoid':
            a = 1 / (1 + np.exp(-x))
        elif activation_type.lower() == 'tanh':
            a = np.tanh(x)
        else:
            a = np.maximum(0, x)
        return a

    @staticmethod
    def derivative_activation(x, activation_type='relu'):
        if activation_type.lower() == 'sigmoid':
            d = np.exp(-x) / ((1 + np.exp(-x))**2)
        elif activation_type.lower() == 'tanh':
            d = 1.0 - np.tanh(x) ** 2
        else:
            d = x
            d[d <= 0] = 0
            d[d > 0] = 1
        return d

    def loss(self, y):
        n, d = self.cache['input'].shape
        n_labels = 10
        labels = np.zeros((n_labels, n))
        if n>1:
            for i in range(n):
                labels[y[i]][i] = 1
        else:
            labels[y] = 1
        self.cache['labels'] = labels
        cost = -np.mean(labels * np.log(self.cache['probabilities']))
        return cost

    def softmax(self, scores):
        scores = np.exp(scores - np.max(scores, axis=0))
        probs = scores / np.sum(scores, axis=0)
        return probs

    def backward(self):
        a1 = self.cache['a1']
        a2 = self.cache['a2']
        h1 = self.cache['h1']
        h2 = self.cache['h2']
        x = self.cache['input']
        assert(self.cache['probabilities'].shape == self.cache['labels'].shape)
        da3 = self.cache['probabilities'] - self.cache['labels']
        dh2 = np.dot(self.parameters['W3'].T, da3)
        dW3 = np.dot(da3, h2.T) / len(x)
        db3 = np.mean(da3, axis=1, keepdims=True)
        da2 = dh2 * self.derivative_activation(a2, activation_type=self.activation_type)
        dh1 = np.dot(self.parameters['W2'].T, da2)
        dW2 = np.dot(da2, h1.T) / len(x)
        db2 = np.mean(da2, axis=1, keepdims=True)
        da1 = dh1 * self.derivative_activation(a1, activation_type=self.activation_type)
        dW1 = np.dot(da1, x) / len(x)
        db1 = np.mean(da1, axis=1, keepdims=True)
        self.grads['b1'] = db1
        self.grads['b2'] = db2
        self.grads['b3'] = db3
        self.grads['W1'] = dW1
        self.grads['W2'] = dW2
        self.grads['W3'] = dW3
        self.update()

    def update(self):
        self.parameters['b1'] -= self.step * self.grads['b1']
        self.parameters['b2'] -= self.step * self.grads['b2']
        self.parameters['b3'] -= self.step * self.grads['b3']
        self.parameters['W1'] -= self.step * self.grads['W1']
        self.parameters['W2'] -= self.step * self.grads['W2']
        self.parameters['W3'] -= self.step * self.grads['W3']

    def train(self):
        losses = []
        self.initialize_weights(initialization_type=self.initialization_type)
        for epoch in range(self.epochs):
            if self.batch_size:
                x, y = sklearn.utils.shuffle(self.training_data, self.training_labels)
                for i in range(0, len(x), self.batch_size):
                    x_batch, y_batch = x[i:i+self.batch_size], y[i:i+self.batch_size]
                    self.forward(x_batch)
                    loss = self.loss(y_batch)
                    losses.append(loss)
                    self.backward()
            else:
                x = self.training_data
                y = self.training_labels
                self.forward(x)
                loss = self.loss(y)
                losses.append(loss)
                self.backward()
            self.forward(x)
            epoch_loss = self.loss(y)
            self.cache['loss'].append(epoch_loss)

    def test(self):
        self.forward(self.test_data)
        probs = self.cache['probabilities']
        predictions = np.argmax(probs, axis=0)
        acc = np.average(self.test_labels == predictions)

        return predictions, acc


if __name__ == '__main__':
    nn = NeuralNetwork(hidden_dims=(1024, 64), activation_type='relu', initialization_type='Glorot', batch_size=64,
                       step=0.05)
    nn.train()
    loss_glorot = nn.cache['loss']
    y_pred_glorot, accuracy = nn.test()
    print('loss: ', loss_glorot)
    print('accuracy: ', accuracy)


