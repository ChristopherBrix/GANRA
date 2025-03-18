from general_attack_model import *


class ManualKissing(GeneralAttackModel):

    def __init__(self, path, script, epsilon, device, l2o_version):
        super().__init__(path, script, epsilon, device, l2o_version)
        self.N = max([int(n.split("_")[-1]) for n in self.inputs]) + 1
        self.M = max([int(n.split("_")[-2]) for n in self.inputs]) + 1

    def forward(self, x):
        x_squared = x * x
        squared_chunks_of_four = x_squared.reshape((x_squared.shape[0], -1, self.N))
        summed_chunks = torch.sum(squared_chunks_of_four, dim=2)
        equalities = self._l2o_eq(summed_chunks - 1)
        all_equalities_satisfied = ((summed_chunks - 1).abs() <= self.epsilon).all(
            dim=1
        )

        x_reshaped = x.view(x.shape[0], self.M, self.N)
        A_indices, B_indices = torch.triu_indices(self.M, self.M, 1)
        rows_A = x_reshaped[:, A_indices]
        rows_B = x_reshaped[:, B_indices]
        squared_diff = (rows_A - rows_B) ** 2
        squared_diff_sum = torch.sum(squared_diff, dim=2)
        inequalities = self._l2o_leq(1 - squared_diff_sum)
        all_inequalities_satisfied = ((1 - squared_diff_sum) <= 0).all(dim=1)

        conditions = torch.concatenate([equalities, inequalities], dim=1)
        all_conditions_satisfied = torch.logical_and(
            all_equalities_satisfied, all_inequalities_satisfied
        )
        return self._l2o_and(conditions), all_conditions_satisfied
