import torch
import torch.nn as nn
from componenets.attention import TemporalAttention, SpatialAttention
from util import reparameterize

class Model(nn.Module):
    def __init__(self, device, num_nodes, input_dim, output_dim, channels, dynamic, lag, horizon, supports,
                 memory_size):
        super(Model, self).__init__()
        self.supports = supports
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.channels = channels
        self.dynamic = dynamic
        self.start_fc = nn.Linear(in_features=input_dim, out_features=self.channels)
        self.memory_size = memory_size

        self.layers = nn.ModuleList(
            [
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=12,
                      cut_size=6, no_proxies=2, memory_size=memory_size),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=3,
                      cut_size=4, no_proxies=2, memory_size=memory_size),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes, cuts=1,
                      cut_size=3, no_proxies=2, memory_size=memory_size),
            ])

        self.skip_layers = nn.ModuleList([
            nn.Linear(in_features=12 * channels, out_features=256),
            nn.Linear(in_features=3 * channels, out_features=256),
            nn.Linear(in_features=1 *channels, out_features=256),
        ])

        self.projections = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, horizon)])

        if self.dynamic:
            self.mu_estimator = nn.Sequential(*[
                nn.Linear(input_dim * lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, memory_size)
            ])

            self.logvar_estimator = nn.Sequential(*[
                nn.Linear(input_dim * lag, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, memory_size)
            ])

    def forward(self, x):
        if self.dynamic:
            mu = self.mu_estimator(x.transpose(3, 1).squeeze())
            logvar = self.logvar_estimator(x.transpose(3, 1).squeeze())
            z_data = reparameterize(mu, logvar)
        else:
            z_data = 0


        x = self.start_fc(x)
        batch_size = x.size(0)

        skip = 0
        for layer, skip_layer in zip(self.layers, self.skip_layers):
            x = layer(x, z_data)
            skip_inp = x.transpose(2, 1).reshape(batch_size, self.num_nodes, -1)
            skip = skip + skip_layer(skip_inp)

        x = torch.relu(skip)

        return self.projections(x).transpose(2, 1).unsqueeze(-1)


class Layer(nn.Module):
    def __init__(self, device, input_dim, num_nodes, cuts, cut_size, dynamic, memory_size, no_proxies):
        super(Layer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.cuts = cuts
        self.cut_size = cut_size
        self.no_proxies = no_proxies
        self.proxies = nn.Parameter(torch.randn(1, cuts * no_proxies, self.num_nodes, input_dim).to(device),
                                    requires_grad=True).to(device)

        self.temporal_att = TemporalAttention(input_dim, num_nodes=num_nodes, cut_size=cut_size)
        self.spatial_att = SpatialAttention(input_dim, num_nodes=num_nodes)

        if self.dynamic:
            self.mu = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)
            self.logvar = nn.Parameter(torch.randn(num_nodes, memory_size).to(device), requires_grad=True).to(device)

        self.temporal_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.spatial_parameter_generators = nn.ModuleList([
            ParameterGenerator(memory_size=memory_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic) for _ in range(2)
        ])

        self.aggregator = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        ])

    def forward(self, x, z_data):
        # x shape: B T N C
        batch_size = x.size(0)

        if self.dynamic:
            z_sample = reparameterize(self.mu, self.logvar)
            z_data = z_data + z_sample

        temporal_parameters = [layer(x, z_data) for layer in self.temporal_parameter_generators]
        spatial_parameters = [layer(x, z_data) for layer in self.spatial_parameter_generators]

        data_concat = []
        out = 0
        for i in range(self.cuts):
            # shape is (B, cut_size, N, C)
            t = x[:, i * self.cut_size:(i + 1) * self.cut_size, :, :]

            proxies = self.proxies[:, i * self.no_proxies: (i + 1) * self.no_proxies]
            proxies = proxies.repeat(batch_size, 1, 1, 1) + out
            t = torch.cat([proxies, t], dim=1)

            out = self.temporal_att(t[:, :self.no_proxies, :, :], t, t, temporal_parameters)
            out = self.spatial_att(out, spatial_parameters)
            out = (self.aggregator(out) * out).sum(1, keepdim=True)
            data_concat.append(out)

        return torch.cat(data_concat, dim=1)

class ParameterGenerator(nn.Module):
    def __init__(self, memory_size, input_dim, output_dim, num_nodes, dynamic):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic

        if self.dynamic:
            print('Using DYNAMIC')
            self.weight_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, input_dim * output_dim)
            ])
            self.bias_generator = nn.Sequential(*[
                nn.Linear(memory_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, output_dim)
            ])
        else:
            print('Using FC')
            self.weights = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
            self.biases = nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def forward(self, x, memory=None):
        if self.dynamic:
            weights = self.weight_generator(memory).view(x.shape[0], self.num_nodes, self.input_dim, self.output_dim)
            biases = self.bias_generator(memory).view(x.shape[0], self.num_nodes, self.output_dim)
        else:
            weights = self.weights
            biases = self.biases
        return weights, biases
