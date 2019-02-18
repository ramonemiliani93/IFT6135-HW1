from neuralnetwork import NeuralNetwork

learning_rates = [0.0005, 0.001, 0.0025, 0.005, 0.025, 0.05]
dimensions = [(1024, 32), (1024, 64), (1024, 128), (512, 512), (512, 256)]
activations = ['relu', 'sigmoid']
best_average_validation_accuracy = 0.00
summaries = []
n = 3
for lr in learning_rates:
    for (dim1, dim2) in dimensions:
        for activation in activations:
            cache = []
            for i in range(n):
                model = NeuralNetwork(hidden_dims=(dim1, dim2), epochs=10, activation_type=activation, step=lr,
                                      batch_size=128)
                model.train()
                _, accuracy = model.test()
                cache.append(accuracy)
            average_validation_accuracy = sum(cache) / n
            model_summary = {'learning_rate': lr, 'hidden_dimensions': (dim1, dim2), 'non_linearity': activation,
                             'average_accuracy': average_validation_accuracy}
            summaries.append(model_summary)
            if average_validation_accuracy > best_average_validation_accuracy:
                best_average_validation_accuracy = average_validation_accuracy
                best_model_summary = model_summary

best = open('best_model.txt', 'w')
best.write(str(best_model_summary))
best.close()
file = open('hp_search.txt', 'w')
for summary in summaries:
    file.write(str(summary))
    file.write('\n')
file.close()





