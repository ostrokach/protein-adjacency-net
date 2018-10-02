# === Basic ===
# import sys

# from .main import main

# sys.exit(main())

# === Profiled ===
from line_profiler import LineProfiler

from pagnn.datavargan import dataset_to_datavar, push_adjs, push_seqs
from pagnn.io import _read_random_row_group, iter_datarows

from .main import main, train
from .utils import basic_permuted_sequence_adder, generate_batch

lp = LineProfiler()
lp.add_function(dataset_to_datavar)
lp.add_function(push_seqs)
lp.add_function(push_adjs)
lp.add_function(iter_datarows)
lp.add_function(train)
lp.add_function(basic_permuted_sequence_adder)
lp.add_function(generate_batch)
lp.add_function(_read_random_row_group)

# Profile the main function
lp_wrapper = lp(main)
lp_wrapper()
# Print results
lp.print_stats()

# ---
# from memory_profiler import profile
# with torch.autograd.profiler.profile() as prof:
#     main()
# print(prof)
# # Add additional functions to profile
# lp.add_function(score_edit)
# lp.add_function(score_blosum62)
# lp.add_function(calculate_statistics_basic)
# lp.add_function(calculate_statistics_extended)
# lp.add_function(evaluate_validation_dataset)
# lp.add_function(evaluate_mutation_dataset)
# # Profile the main function
# lp_wrapper = lp(main)
# lp_wrapper()
# # Print results
# lp.print_stats()
