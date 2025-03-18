from general_attack_model import *


class Sequential(GeneralAttackModel):

    def __init__(self, path, script, epsilon, device, l2o_version):
        super().__init__(path, script, epsilon, device, l2o_version)
        self.smt_form = utils.extract_formula(script)
        if self.smt_form[0] == "and":
            formulas = self.smt_form[1:]
        else:
            assert (
                self.smt_form[0] == "="
            ), "Logic is not very modular and only supports mbo/kissing. Extending it should be straightforward."
            formulas = [self.smt_form]
        self.sequential_computations = []
        for formula in formulas:
            assert formula[0] in ["=", "<=", "<"]
            assert formula[2] == 0
            self.sequential_computations.append(
                (formula[0], self._build_node(formula, device=device))
            )

    def forward(self, x):
        all_constraints_satisfied = True
        individual_constraints = []
        for constraint_type, sequential_computation in self.sequential_computations:
            res = sequential_computation(x)
            if constraint_type == "=":
                individual_constraints.append(self._l2o_eq(res))
                all_constraints_satisfied &= res.abs() <= self.epsilon
            elif constraint_type == "<=":
                individual_constraints.append(self._l2o_leq(res))
                all_constraints_satisfied &= res <= 0
            else:
                assert constraint_type == "<"
                individual_constraints.append(self._l2o_lt(res))
                all_constraints_satisfied &= res < 0
        conditions = torch.stack(individual_constraints, dim=1)
        return self._l2o_and(conditions), all_constraints_satisfied
