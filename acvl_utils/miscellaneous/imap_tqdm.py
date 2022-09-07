from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def imap_tqdm(function, iterable, processes, chunksize=1, **kwargs):
    """
    Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
    Results are always ordered and the performance should be the same as of Pool.map.
    TODO: Still needs more performance testing
    :param function: The function that should be parallelized.
    :param iterable: The iterable passed to the function.
    :param processes: The number of processes used for the parallelization.
    :param kwargs: Any additional arguments that should be passed to the function.
    """
    if kwargs:
        function_wrapper = partial(wrapper, function=function, **kwargs)
    else:
        function_wrapper = partial(wrapper, function=function)

    results = [None] * len(iterable)
    with Pool(processes=processes) as p:
        with tqdm(total=len(iterable)) as pbar:
            for i, result in p.imap_unordered(function_wrapper, enumerate(iterable), chunksize=chunksize):
                results[i] = result
                pbar.update()
    return results


def wrapper(enum_iterable, function, **kwargs):
    i = enum_iterable[0]
    result = function(enum_iterable[1], **kwargs)
    return i, result