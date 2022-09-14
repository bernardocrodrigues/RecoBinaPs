from torch import device, zeros, no_grad, squeeze
from torch.nn import init, Module
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from alive_progress import alive_it  # dynamic progress bars

from .loss import WeightedXor
from .layers import BinarizeTensorThresh, BinaryActivation, EncoderModule, DecoderModule


class Net(Module):
    def __init__(self, init_weights, tensor_device):
        super(Net, self).__init__()

        input_dim = init_weights.size()[1]
        hidden_dim = init_weights.size()[0]

        self.encoder = EncoderModule(input_dim,
                                     hidden_dim,
                                     init_weights,
                                     tensor_device)

        self.decoder = DecoderModule(hidden_dim,
                                     input_dim,
                                     self.encoder.weight.data,
                                     self.encoder.weightB.data,
                                     tensor_device)

        self.activation_encoder = BinaryActivation(hidden_dim,
                                                   tensor_device)
        self.activation_decoder = BinaryActivation(input_dim,
                                                   tensor_device)

        self.clip_weights()
        self.to(tensor_device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation_encoder(x, False)
        x = self.decoder(x)
        return self.activation_decoder(x, True)

    def clip_weights(self, mini=-1, maxi=1):
        self.encoder.clipWeights(mini, maxi)
        self.activation_encoder.clip_bias()
        self.activation_decoder.no_bias()

    def train_model(self, train_loader, optimizer, loss_function):
        self.train()
        for data, _ in train_loader:
            optimizer.zero_grad()
            output = self(data)
            iterator_encoder_weights = [
                par for name, par in self.named_parameters() if name.endswith("encoder.weight")]
            loss = loss_function(output, data, next(
                iter(iterator_encoder_weights)))
            loss.backward()
            optimizer.step()
            self.clip_weights()

    def test_model(self, test_loader, loss_function):
        self.eval()
        test_loss = 0
        correct = 0
        with no_grad():
            for data, _ in test_loader:
                output = self(data)
                iterator_encoder_weights = [par for name, par in self.named_parameters() if name.endswith("encoder.weight")]
                test_loss += loss_function(output, data, next(iter(iterator_encoder_weights)))
                correct += (output.ne(data.data.view_as(output)).sum(1) == 0).sum()
        next(iter(test_loader))

        return test_loss, correct


def learn(train_dataset, test_dataset, batch_size, test_batch_size, hidden_dimension, lr, weight_decay, gamma, epochs):

    device_gpu = device("cuda")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True)

    weights = zeros(hidden_dimension, train_dataset.ncol(), device=device_gpu)
    init.constant_(weights, 0)
    weights.clamp_(1/(train_dataset.ncol()), 1)

    bias_init = zeros(hidden_dimension, device=device_gpu)
    init.constant_(bias_init, -1)

    model = Net(weights, device_gpu)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = WeightedXor(train_dataset.getSparsity(), weight_decay)
    scheduler = MultiStepLR(optimizer, [5, 7], gamma=gamma)

    for current_epoch in alive_it(range(1, epochs + 1), force_tty=True):
        model.train_model(train_loader, optimizer, loss_function)
        test_loss, correct = model.test_model(test_loader, loss_function)
        scheduler.step()

        if current_epoch % 10 == 0:
            print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


    return model, weights


def get_patterns(weights):
    patterns = []
    with no_grad():
        for hn in BinarizeTensorThresh(weights, .2):
            pat = squeeze(hn.nonzero())
            patterns.append(pat.cpu().numpy())
    return patterns
