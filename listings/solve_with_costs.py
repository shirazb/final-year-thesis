# Will profile then solve
policy = chkpter.solve_optimal_policy(M)

# Will reuse the profiling results
policy2 = chkpter.solve_optimal_policy(another_M)

# Will use uniform costs
policy3 = chkpter.solve_optimal_policy(M, profile_mem=False, profile_comp=False)

# Will profile compute and use given memory costs
policy4 = chkpter.solve_optimal_policy(M, profile_mem=False, memory_costs=costs)