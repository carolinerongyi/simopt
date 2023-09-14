import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ..base import Solution, Solver
class NelderMead(Solver):
    def __init__(self, name="GASSO", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}

        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box" # ??
        self.variable_type = "continuous"
        self.gradient_needed = True

        