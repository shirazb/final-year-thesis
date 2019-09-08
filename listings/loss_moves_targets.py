class LossLayer(torch.nn.Module):
    def __init__(self, loss, targets_cpu, device):
        self.loss = loss
        self.targets_cpu = targets_cpu
        self.device = device
    
    def forward(self, x):
        return self.loss(x, self.targets_cpu.to(self.device))