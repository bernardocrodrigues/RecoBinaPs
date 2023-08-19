# Copyright 2021 Jonas Fischer <fischer@mpi-inf.mpg.de>
# Copyright 2022 Bernardo C. Rodrigues
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
# have received a copy of the GNU General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# This code was based on binaps/Binaps_code/my_layers.py

import torch
import torch.nn as nn
from torch.autograd import Function


def BinarizeTensorThresh(tens, thresh):
    return (tens > thresh).float()


def BinarizeTensorStoch(tens):
    return tens.bernoulli()


def SteppyBias(tensor, is_dec):
    if is_dec:
        t = tensor.clamp(min=0, max=0)
    else:
        t = tensor.clamp(max=-1)
    return t.int().float()


class BinarizeFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, biasNI):
        ctx.save_for_backward(input, bias, biasNI)
        res = (input+biasNI).clamp(0, 1).round()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, biasNI = ctx.saved_tensors
        if (bias[0] < 0):
            grad_input = (input+biasNI).clamp(0, 1).round()*grad_output
        else:
            grad_input = grad_output
        # Throw out negative gradient to bias if node was not active
        # (Do not punish for things it did not do)
        grad_bias = (1-(input+biasNI).clamp(0, 1).round())*grad_output.clamp(
            max=0).sum(0) + (input+biasNI).clamp(0, 1).round()*grad_output.sum(0)
        return grad_input, grad_bias, None


class BinaryActivation(nn.Module):

    def __init__(self, size, device_gpu):
        super(BinaryActivation, self).__init__()
        self.bias = nn.Parameter(-torch.ones(size,
                                 device=device_gpu), requires_grad=True)
        self.biasNI = self.bias.clone().detach().to(device_gpu)

    def forward(self, input, is_dec):
        with torch.no_grad():
            self.biasNI = SteppyBias(self.bias, is_dec)
        return BinarizeFunction.apply(input, self.bias, self.biasNI)

    def clip_bias(self):
        with torch.no_grad():
            self.bias.clamp_(max=-1)

    def no_bias(self):
        with torch.no_grad():
            self.bias.clamp_(min=0, max=0)


class BinarizedLinearModule(nn.Module):

    class BinaryLinearFunction(Function):
        @staticmethod
        def forward(context, input_data, weight_a, weight_b):
            context.save_for_backward(input_data, weight_a, weight_b)
            out = input_data.matmul(weight_b.t())
            return out

        @staticmethod
        def backward(ctx, grad_output):
            input_data, weight, _ = ctx.saved_tensors
            grad_input = grad_output.matmul(weight)
            grad_weight = grad_output.t().matmul(input_data)
            return grad_input, grad_weight, None

    def __init__(self, input_size):
        super(BinarizedLinearModule, self).__init__()
        self.input_size = input_size

    def clipWeights(self, mini=-1, maxi=1):
        with torch.no_grad():
            self.weight.clamp_(1/(self.input_size), maxi)


class EncoderModule(BinarizedLinearModule):

    def __init__(self, input_size, output_size, enc_weights, device):

        super(EncoderModule, self).__init__(input_size)

        self.weight = nn.Parameter(enc_weights)
        self.weightB = torch.zeros(output_size, input_size, device=device)

    def forward(self, input_data):
        with torch.no_grad():
            self.weightB.data.copy_(BinarizeTensorStoch(self.weight))

        return self.BinaryLinearFunction.apply(input_data, self.weight, self.weightB)


class DecoderModule(BinarizedLinearModule):

    def __init__(self, input_size, output_size, enc_weights, enc_weights_b, device):

        super(DecoderModule, self).__init__(input_size)

        self.weight = nn.Parameter(torch.zeros(
            input_size, output_size, device=device))
        self.weightB = torch.zeros(input_size, output_size, device=device)
        self.weight.data = enc_weights.transpose(0, 1)
        self.weightB.data = enc_weights_b.transpose(0, 1)

    def forward(self, input_data):
        return self.BinaryLinearFunction.apply(input_data, self.weight, self.weightB)
