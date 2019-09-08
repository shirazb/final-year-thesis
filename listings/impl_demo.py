sequence = # ... a torch.nn.Sequential or a list of layers

chkpter = SequentialCheckpointer(sequence)

chkpter.profile_sequence(dummy_input, dummy_upstream_grad)
policy = chkpter.solve_optimal_policy(memory_budget)

# Inside training loop
downstream_grad = chkpter.backprop_sequence(policy, x, upstream_grad)