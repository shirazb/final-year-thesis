x = torch.randn(10, 10)
y = torch.randn(10, 10)
x.requires_grad = True
y.requires_grad = True

z = x * y