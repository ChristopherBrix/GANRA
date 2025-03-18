from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import utils
import numpy as np


class GeneralAttackModel(nn.Module, ABC):

    @abstractmethod
    def __init__(self, path, script, epsilon, device, l2o_version):
        super(GeneralAttackModel, self).__init__()
        self.epsilon = epsilon
        _, self.inputs = utils._extract_input(script)
        self.l2o_version = l2o_version
        self.variables = utils.get_variables(script)

    @abstractmethod
    def forward(self, x):
        pass

    def _l2o_eq(self, x):
        if self.l2o_version == "ours":
            return torch.clip(torch.abs(x) - self.epsilon, min=0)
        elif self.l2o_version in ["ugotnl", "nrago"]:
            return x**2
        else:
            raise ValueError(f"Unknown L2O version {self.l2o_version}")

    def _l2o_leq(self, x):
        if self.l2o_version == "ours":
            return torch.clip(x, min=0)
        elif self.l2o_version == "ugotnl":
            return torch.where(x <= 0, 0, x**2)
        elif self.l2o_version == "nrago":
            return x
        else:
            raise ValueError(f"Unknown L2O version {self.l2o_version}")

    def _l2o_lt(self, x):
        if self.l2o_version == "ours":
            return torch.clip(x, min=-self.epsilon)
        elif self.l2o_version == "ugotnl":
            return self._l2o_leq(x)
        elif self.l2o_version == "nrago":
            return x
        else:
            raise ValueError(f"Unknown L2O version {self.l2o_version}")

    def _l2o_and(self, x):
        assert x.ndim == 2
        if self.l2o_version in ["ours", "ugotnl"]:
            return x.sum(dim=1)
        elif self.l2o_version == "nrago":
            return (torch.clip(x, min=0)).sum(dim=1)
        else:
            raise ValueError(f"Unknown L2O version {self.l2o_version}")

    def _build_node(self, node, device):
        if isinstance(node, bool):
            return lambda _: torch.tensor(node, dtype=torch.bool, device=device)
        elif isinstance(node, str):  # occurence of variable
            idx = self.inputs.index(node)
            return lambda assignments: assignments[:, idx]
        elif isinstance(node, float) or isinstance(node, int):
            return lambda assignments: torch.full(
                (assignments.shape[0],), node, dtype=torch.float32, device=device
            )
        elif node[0] == "and":
            children = [self._build_node(arg, device=device) for arg in node[1:]]
            return lambda assignments: self._l2o_and(
                torch.stack([child(assignments) for child in children]),
            )
        elif node[0] == "or":
            children = [self._build_node(arg, device=device) for arg in node[1:]]
            raise NotImplementedError("OR operation not yet implemented.")
        # Comparison operators
        elif node[0] == "=":
            left = self._build_node(node[1], device=device)
            assert node[2] == 0
            return lambda assignments: self._l2o_eq(left(assignments))
        elif node[0] == "<=":
            left = self._build_node(node[1], device=device)
            assert node[2] == 0
            return lambda assignments: self._l2o_leq(left(assignments))
        elif node[0] == "<":
            left = self._build_node(node[1], device=device)
            assert node[2] == 0
            return lambda assignments: self._l2o_lt(left(assignments))
        elif node[0] == ">":
            print("Should not happen, > occured.", flush=True)
            raise TypeError
        elif node[0] == ">=":
            print("Should not happen, >= occured.", flush=True)
            raise TypeError
        # Arithmetic operators
        elif node[0] == "+":
            children = [self._build_node(arg, device=device) for arg in node[1:]]
            return lambda assignments: torch.sum(
                torch.stack([child(assignments) for child in children]),
                dim=0,
            )
        elif node[0] == "-":
            left = self._build_node(node[1], device=device)
            right = self._build_node(node[2], device=device)
            return lambda assignments: torch.subtract(
                left(assignments), right(assignments)
            )
        elif node[0] == "*":
            children = [self._build_node(arg, device=device) for arg in node[1:]]
            return lambda assignments: torch.prod(
                torch.stack([child(assignments) for child in children]),
                dim=0,
            )
        else:
            raise ValueError(f"Unsupported operation: {node[0]}")

    def traverse_summands_of_mbo(self, path):
        """
        Extracts all relevant information of the mbo formula from the file.

        Input: The path to the mbo file.
        Output: One list that contains the constant factor used for multiplying in every summand, another list that contains the factors in each summand in a string format.
        """
        with open(path, "r") as file:
            assertions = file.read().split("(assert")[1]
        multiplications = assertions.split("(+")[1]
        summands = []
        stack = []
        start_idx = None
        scalar_sums = []
        multiplications = multiplications.split()

        for index, chunk in enumerate(
            multiplications
        ):  # chunks are either (-, (*, floats, variables, or variables)
            if chunk == "(-":
                if not stack:
                    start_idx = index  # start of new summand
                    scalar_sums.append(1.0)
                stack.append("(")
                scalar_sums[-1] = -1 * scalar_sums[-1]
            elif chunk == "(*":
                if not stack:
                    start_idx = index  # start of new summand
                    scalar_sums.append(1.0)
                stack.append("(")
            elif chunk.isdigit():
                scalar_sums[-1] = scalar_sums[-1] * float(chunk)
            elif ")" in chunk:
                for _, char in enumerate(
                    chunk
                ):  # needs to be traversed one by one, consists of a variable and possibly multiple ')'
                    if char == ")":
                        if not stack:
                            break
                        stack.pop()

                        if not stack:  # stack got emptied
                            if (
                                start_idx is None
                            ):  # one more close parenthesis than open ones seen, final end
                                return scalar_sums, summands
                            else:  # stack got emptied and start_idx is given -> end of summand
                                summands.append(
                                    "".join(
                                        multiplications[start_idx : index + 1]
                                    ).strip()
                                )
                                start_idx = None  # Reset for the next expression

        return scalar_sums, summands

    def get_summands_of_mbo(self, path, device, is_preprocessed_mbo):
        scalar_sums, summands = self.traverse_summands_of_mbo(path)
        if is_preprocessed_mbo:
            scalar_sums = scalar_sums + scalar_sums
            summands = summands + [
                x.replace("p", "n") for x in summands
            ]  # add negative variables
        assert len(summands) == len(scalar_sums)

        occurences = np.array(
            [[txt.count(variable) for txt in summands] for variable in self.variables]
        )

        max_power = [max(occurences[index]) + 1 for index in range(len(self.variables))]
        if is_preprocessed_mbo:
            max_power[1::2] = max_power[0::2]

        sum_index = np.array(
            [
                [
                    sum(max_power[0:var_ind]) + txt.count(self.variables[var_ind])
                    for txt in summands
                ]
                for var_ind in range(len(self.variables))
            ]
        )  # same dimension as occurences but with index shift for all var_ind > 0

        scalars = torch.tensor(scalar_sums, device=device)
        return max_power, sum_index, scalars

    def _ensure_tensor(self, res):
        if not isinstance(res, torch.Tensor):
            res = torch.tensor(res, dtype=torch.float32)
        return res

    def get_index_of_input(self, name):
        for ind in range(len(self.inputs)):
            if self.inputs[ind] == name:
                return ind
            if self.inputs[ind] == name:
                return ind
        raise NameError

    def get_input_of_index(self, index):
        return self.inputs[index]

    def print_inputs(self):
        for ind in range(len(self.inputs)):
            print("Index: ", ind, "   ", self.inputs[ind], flush=True)

    def get_dimension_input(self):
        return len(self.inputs)
