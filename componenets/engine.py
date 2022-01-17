import torch
import torch.optim as optim

from componenets.model import Model


class trainer():
    def __init__(self, scaler, device, lrate, num_nodes, input_dim, output_dim, channels,
                 grad_norm, dynamic, lag, horizon, supports, memory_size):
        self.model = Model(device=device, num_nodes=num_nodes, input_dim=input_dim, output_dim=output_dim,
                           channels=channels, dynamic=dynamic, lag=lag, horizon=horizon, supports=supports,
                           memory_size=memory_size)
        self.model.to(device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lrate)
        self.loss = torch.nn.SmoothL1Loss()
        self.scaler = scaler
        self.clip = 5
        self.grad_norm = grad_norm
        self.dynamic = dynamic
        self.device = device

    def train(self, input, real_val, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        label = self.scaler.inverse_transform(real_val)
        output = self.scaler.inverse_transform(output)

        # prediction loss
        prediction = self.loss(output, label)
        if self.dynamic:
            mu = []
            logvar = []
            for layer in self.model.layers:
                mu.append(layer.mu)
                logvar.append(layer.logvar)

            logvar = torch.stack(logvar)
            mu = torch.stack(mu)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            mu = self.model.mu_estimator(input.transpose(3, 1).squeeze())
            logvar = self.model.logvar_estimator(input.transpose(3, 1).squeeze())

            data_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            KLD = KLD + data_KLD
        else:
            KLD = torch.Tensor([0.]).to(self.device)

        loss = prediction + 0.0005 * KLD
        loss.backward()

        # add max grad clipping
        if self.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return prediction.item(), KLD.item(), 0.

    def eval(self, input, real_val):
        output = self.model(input)
        label = self.scaler.inverse_transform(real_val)
        output = self.scaler.inverse_transform(output)

        loss = self.loss(output, label)
        return loss.item()
