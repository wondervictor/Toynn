"""
Gradient Check
"""

import torch
import torch.nn as nn
import numpy as np

import loss
import layer


def test_square_loss():

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



if __name__ == '__main__':
    test_fully_connected()
