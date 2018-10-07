"""
Gradient Check
"""

import torch
import torch.nn as nn
import numpy as np

import loss
import layer
import activation


def test_square_loss():
    print("gradient check: MSE")

    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    square_loss_func_torch = nn.MSELoss()
    square_loss_func = loss.SquareLoss()

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    square_loss_torch = square_loss_func_torch(torch_x, torch.Tensor(y))
    square_loss = square_loss_func(x, y)
    print("Value:\ntorch:{},mine:{}, delta:{}"
          .format(square_loss_torch.item(), square_loss, (square_loss-square_loss_torch.item())))
    square_loss_torch.backward()
    torch_x_grad = torch_x.grad.data.numpy()
    x_grad = square_loss_func.backward()
    print(np.sum(x_grad - torch_x_grad, 0))


def test_fully_connected():
    print("gradient check: FullyConnected")

    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*10).reshape((5, 10)).astype('float32')

    # --- mine --
    fc = layer.FullyConnected('train', 8, 10)
    sqaure_loss_func = loss.SquareLoss()
    fc_out = fc(x)
    sqaure_loss = sqaure_loss_func(fc_out, y)

    # --- torch ---
    weights = fc.weights.get_data()
    bias = fc.bias.get_data()

    torch_fc = nn.Linear(8, 10)
    torch_fc.weight.data.copy_(torch.Tensor(weights.T))
    torch_fc.bias.data.copy_(torch.Tensor(bias))
    torch_square_func = nn.MSELoss()
    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    torch_fc_out = torch_fc(torch_x)
    torch_sqaure_loss = torch_square_func(torch_fc_out, torch.Tensor(y))

    print("Value:\ntorch:{}, mini:{}, delta:{}".format(
        torch_sqaure_loss.item(), sqaure_loss, (torch_sqaure_loss.item()-sqaure_loss)
    ))

    # --- my grad ---
    grad_x = sqaure_loss_func.backward()
    grad_fc = fc.backward(grad_x)
    grad_w = fc.weights.get_grad()
    grad_b = fc.bias.get_grad()

    # --- torch grad ---
    torch_sqaure_loss.backward()
    torch_grad_x = torch_x.grad.data.numpy()
    torch_grad_w = torch_fc.weight.grad.data.numpy()
    torch_grad_b = torch_fc.bias.grad.data.numpy()

    print("--grad x ---")
    print(grad_fc-torch_grad_x)

    print("--grad w ---")
    print(grad_w-torch_grad_w.T)

    print("--grad b ---")
    print(grad_b-torch_grad_b)


def test_softmax():
    print("gradient check: Softmax")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    softmax = activation.Softmax()
    sqaure_loss_func = loss.SquareLoss()

    softmax_x = softmax(x)
    square_loss = sqaure_loss_func(softmax_x, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    softmax_torch = nn.Softmax()
    square_loss_func_torch = nn.MSELoss()
    softmax_x_torch = softmax_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(softmax_x_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_softmax = sqaure_loss_func.backward()
    grad_x = softmax.backward(grad_softmax)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_sigmoid():
    print("gradient check: Sigmoid")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    sigmoid = activation.Sigmoid()
    sqaure_loss_func = loss.SquareLoss()

    sigmoid_x = sigmoid(x)
    square_loss = sqaure_loss_func(sigmoid_x, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    sigmoid_torch = nn.Sigmoid()
    square_loss_func_torch = nn.MSELoss()
    sigmoid_x_torch = sigmoid_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(sigmoid_x_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = sigmoid.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_relu():
    print("gradient check: ReLU")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    act = activation.ReLU()
    sqaure_loss_func = loss.SquareLoss()

    y_ = act(x)
    square_loss = sqaure_loss_func(y_, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    act_torch = nn.ReLU()
    square_loss_func_torch = nn.MSELoss()
    y_torch = act_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(y_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = act.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_leakyrelu():
    print("gradient check: Leaky ReLU")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    act = activation.LeakyReLU(negative_slope=0.4)
    sqaure_loss_func = loss.SquareLoss()

    y_ = act(x)
    square_loss = sqaure_loss_func(y_, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    act_torch = nn.LeakyReLU(negative_slope=0.4)
    square_loss_func_torch = nn.MSELoss()
    y_torch = act_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(y_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = act.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_elu():

    print("gradient check: ELU")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    act = activation.ELU(alpha=0.2)
    sqaure_loss_func = loss.SquareLoss()

    y_ = act(x)
    square_loss = sqaure_loss_func(y_, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    act_torch = nn.ELU(alpha=0.2)
    square_loss_func_torch = nn.MSELoss()
    y_torch = act_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(y_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = act.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_tanh():

    print("gradient check: Tanh")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    act = activation.Tanh()
    sqaure_loss_func = loss.SquareLoss()

    y_ = act(x)
    square_loss = sqaure_loss_func(y_, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    act_torch = nn.Tanh()
    square_loss_func_torch = nn.MSELoss()
    y_torch = act_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(y_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = act.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


if __name__ == '__main__':
    test_leakyrelu()
    test_elu()
    test_tanh()
