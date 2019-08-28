import numpy as np
import torch

# Numpy like API with derivatives and GPU acceleration.
a = torch.Tensor([[1., 2.], [-1., 3.]], device='cuda')
a.requires_grad = True

b = torch.randn(2, 2, requires_grad=True).cuda()

c = a + b
d = torch.full_like(a, 15., requires_grad=True)

# From numpy.
e_np = np.ones((2, 2), dtype=np.single)
e = torch.from_numpy(e_np).to('cuda')

# Broadcasting.
f = ((e + 2) ** 1.3) * d

# ML functions.
g = torch.nn.functional.relu(c)
h = torch.nn.functional.linear(f, g).sum()

# Perform backward pass.
h.backward()

a.grad
  # ----> The gradient tensor.
b.grad
  # ----> The gradient tensor.
c.grad
  # ----> None, c is not a leaf.
d.grad
  # ----> The gradient tensor.
e.grad
  # ----> None, did not require gradient.
