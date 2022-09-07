from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def imap_tqdm(function, iterable, processes, ordered=True, **kwargs):
    """
    Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
    By default, the result has the same ordering as the iterable. This might slow down the parallelization under certain circumstances.
    The results can also be returned unordered, which should be equally fast as Pool.map.
    :param function: The function that should be parallelized.
    :param iterable: The iterable passed to the function.
    :param processes: The number of processes used for the parallelization.
    :param ordered: If results are ordered or unordered.
    :param kwargs: Any additional arguments that should be passed to the function.
    """
    if kwargs:
        function = partial(function, **kwargs)

    results = []
    with Pool(processes=processes) as p:
        with tqdm(total=len(iterable)) as pbar:
            if ordered:
                imap = p.imap
            else:
                imap = p.imap_unordered
            for result in imap(function, iterable):
                results.append(result)
                pbar.update()
    return results