import os
from glob import glob
from time import time
from typing import Callable, List

from owlready2 import get_ontology


def exec_timer(name: str = ""):
    def dec_inner(func: Callable):
        def dec_inner_inner(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            print(f"{name} Took {(time() - start):.3f} seconds")
            print("==========================================================")

            return result

        return dec_inner_inner

    return dec_inner

def split_iterable_into_batches(elements_list: List, batch_size: int):
    return [
        elements_list[element: element + batch_size]
        for element in range(0, len(elements_list), batch_size)
    ]

def load_ontologies(path: str, extension: str = "owl") -> List:
    """
    A function to load ontologies based on a file extension
    :param path: base directory for ontologies.
    :param extension: ontologies extension.
    :return:
    """
    extension = extension.replace(".", "")
    ontologies_dirs = glob(f"{os.path.join(path, f'*.{extension}')}")
    ontologies = []

    for ontology_dir in ontologies_dirs:
        ontologies.append(get_ontology(ontology_dir).load())

    return ontologies
