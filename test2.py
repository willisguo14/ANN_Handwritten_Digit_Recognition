import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 1, 10, 0.5, lmbda=5.0, evaluation_data=validation_data, early_stopping_n = 2)