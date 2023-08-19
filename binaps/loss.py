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
# This code was based on binaps/Binaps_code/my_loss.py

from torch.nn import Module, ReLU


class WeightedXor(Module):

    def __init__(self, weight, weight_decay):
        super(WeightedXor, self).__init__()
        self.weight = weight
        self.weight_decay = weight_decay

    def forward(self, output, target, w):

        relu = ReLU()

        diff = relu((output - target)).sum(1).mul(self.weight).mean() + \
               relu((target - output)).sum(1).mul(1-self.weight).mean()
        diff += self.weight_decay * \
                (((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2).sum())

        return diff
