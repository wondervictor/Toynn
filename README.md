# Toynn
Toy Neural Neworks from scratch


### Features

* Interface designed is pretty simple, even simpler than Keras
* Implemented with Pure Python and Numpy
* All operators are vectorized
* Easy to support more **Layers/Operators**


### Operators

* Layer: FullyConnected, BatchNorm, Dropout
* Activation: ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Softmax
* Loss: L1 Loss, Cross Entropy, L2 Loss
* Optimizer: SGD, Momentum, Adam

### Documentation

##### 1. Define Neural Network

````python

# model
model = network.Network()

# optional: define intializer for weights 
initializer = parameter.GaussianInitializer(std=0.1)
bias_initializer = parameter.ConstantInitializer(0.1)

# add layers
model.add(
    FullyConnected(
        name='fc1',
        in_feature=784,
        out_feature=512,
        weight_initializer=initializer,
        bias_initializer=bias_initializer))
model.add(
    BatchNorm(name='bn1',
              num_features=512)
)
model.add(ReLU(name='relu1'))

````

##### 2. Loss, Optimizer

````python

# define loss function
model.add_loss(CrossEntropyLoss())

# define optimizer
optimizer = Momentum(lr=lr, momentum=0.9)

````

##### 3. Provide Data and Training

````python

# provide data
batch_images = ...
batch_labels = ...

# train
output, loss = model.forward(batch_images, batch_labels)
model.optimize(optimizer)

````

##### 4. When testing

````python
# to freeze BatchNorm or Dropout
model.test_mode()

````


### Licence

This project is under the **MIT Licence**