import sys

from pagnn import settings
from pagnn.training.dcn.main import main

if settings.PROFILER == "cProfile":
    import cProfile

    sys.exit(cProfile.run("main()", filename="parent.prof"))
elif settings.PROFILER == "line_profiler":
    from line_profiler import LineProfiler

    from pagnn.datavargan import dataset_to_datavar, gen_adj_pool, push_adjs, push_seqs
    from pagnn.io import _read_random_row_group, iter_datarows

    from pagnn.training.dcn.main import train
    from pagnn.training.dcn.utils import basic_permuted_sequence_adder, generate_batch

    lp = LineProfiler()
    lp.add_function(dataset_to_datavar)
    lp.add_function(gen_adj_pool)
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
else:
    sys.exit(main())
