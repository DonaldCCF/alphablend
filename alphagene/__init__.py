__all__ = ['GeneticOptimizer',
           'make_function',
           'function_map']

__version__ = '1.0.0'

from ._gene import GeneticOptimizer
from ._func import make_function
from ._map import function_map
