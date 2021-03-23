import sys

from .main import main

# === Basic ===
sys.exit(main())

# === Profiled ===
# with torch.autograd.profiler.profile() as prof:
#     main()
# print(prof)
# from memory_profiler import profile
# from line_profiler import LineProfiler
# lp = LineProfiler()
# # Add additional functions to profile
# lp.add_function(score_edit)
# lp.add_function(score_blosum62)
# lp.add_function(calculate_statistics_basic)
# lp.add_function(calculate_statistics_extended)
# lp.add_function(evaluate_validation_dataset)
# lp.add_function(evaluate_mutation_dataset)
# lp.add_function(train)
# # Profile the main function
# lp_wrapper = lp(main)
# lp_wrapper()
# # Print results
# lp.print_stats()
