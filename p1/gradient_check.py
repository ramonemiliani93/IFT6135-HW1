from neuralnetwork import NeuralNetwork, X_train, y_train
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)

nn = NeuralNetwork((1024, 64), epochs=10, activation_type='relu', batch_size=None, training_data=X_train[0],
                   training_labels=y_train[0])
nn.train()


def gradient_check(model, x, y, theta, N):
    eps = 1 / N
    p = 10
    nn = model
    parameter = nn.parameters[theta]
    rows, cols = parameter.shape
    differences = []
    max_diff = -1
    nn.forward(x)
    true_value = nn.loss(y)
    true_grad = nn.grads[theta]
    if cols > 1:
        for i in range(p):
            for a in range(cols):
                nn.parameters[theta] = parameter
                nn.forward(x)
                v1 = nn.loss(y)
                assert (true_value == v1)

                nn.parameters[theta][i][a] += eps
                nn.forward(x)
                loss1 = nn.loss(y)
                nn.parameters[theta][i][a] = parameter[i][a]
                nn.forward(x)
                nn.parameters[theta][i][a] -= eps
                nn.forward(x)
                loss2 = nn.loss(y)
                finite_difference = (loss1 - loss2) / (2 * eps)
                diff = abs(finite_difference - true_grad[i][a])

                differences.append(diff)
                if diff > max_diff:
                    print('finite_difference: ', finite_difference)
                    print('true gradient: ', nn.grads[theta][i][a])
                    max_diff = diff

    max_diff = max(differences)
    return max_diff


N_values = [100, 1000, 5000, 7000, 10000, 20000]
vals = []
_x = X_train[0]
_y = y_train[0]
for N in N_values:
    val = gradient_check(nn, _x, _y, theta='W2', N=N)
    vals.append(val)

print(vals)
plt.plot(N_values, vals)
plt.ylabel('Maximum Absolute Difference')
plt.xlabel('N')
plt.show()
print(vals)

