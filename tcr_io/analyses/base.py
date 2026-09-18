### Analyses: 
# goal: make frequently used analyses easily runnable on new datasets
# difference with operations : operations are low-level, do not require parameters or tuning, can be used as building blocks for analyses. 
# analyses: use metadata columns, can be run with different parameters, can be used to generate figures and tables for publications, etc.
# operations can be rebuilt from scratch, analyses are more like a "recipe" that uses operations to produce a specific output. 



from abc import ABC, abstractmethod
from typing import ClassVar, List
from ..structure import Artifact, Store
from ..expressions import KNOWN_LOCI
from ..operations import BaseOperation

class BaseAnalysis(ABC):
    name : str = "base_analysis"
    version : str = "0.0"
    description : str = "Base analysis - does nothing"
    supported_loci : ClassVar[frozenset] = KNOWN_LOCI # analyses are not locus-specific by default, but can be if needed.
    required_operations : List[BaseOperation] = [] # analyses can require specific operations to be run before them, to ensure the necessary data is available.
    

# analyses to implement:
# fisher exact (associated clones)
# expansion/contraction test (binomial t-test, timepoint comparison -> stable, expanding, expanding and new, contracting, with BH correction)