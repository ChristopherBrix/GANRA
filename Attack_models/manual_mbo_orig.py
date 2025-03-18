from general_attack_model import *


class ManualMboOrig(GeneralAttackModel):

    def __init__(self, path, script, epsilon, device, l2o_version):
        super().__init__(path, script, epsilon, device, l2o_version)
        self.max_power, sum_index, self.scalars = self.get_summands_of_mbo(
            path, device=device, is_preprocessed_mbo=False
        )
        self.sum_index = torch.tensor(sum_index, device=device)

        self.base_indices = []
        self.exponents = []
        for var_ind in range(len(self.max_power)):
            self.base_indices += [var_ind] * self.max_power[var_ind]
            self.exponents += list(range(self.max_power[var_ind]))
        self.exponents = torch.tensor(self.exponents, device=device)[None, :]

    def forward(self, x):
        batch_size, num_vars = x.shape
        x_broadcasted = x[:, self.base_indices]
        precomps = x_broadcasted**self.exponents

        res = (
            self.scalars.detach().clone().unsqueeze(0).expand(batch_size, -1)
        )  # to get shape (batch_size, num_vars )

        for var_index in self.sum_index:
            indices = var_index.unsqueeze(0).expand(batch_size, -1)
            factors = torch.gather(
                precomps, dim=1, index=indices
            )  # Shape: [batch_size, num_indices]
            res = torch.mul(res, factors)

        res_summed = torch.sum(res, dim=1)
        res_is_zero = self._l2o_eq(res_summed)
        res_is_zero_satisfied = res_summed == 0

        pos_constr = self._l2o_lt(-x)
        pos_constr_satisfied = (-x < 0).all(dim=1)

        conditions = torch.cat([res_is_zero.unsqueeze(1), pos_constr], dim=1)
        all_conditions_satisfied = res_is_zero_satisfied & pos_constr_satisfied
        return self._l2o_and(conditions), all_conditions_satisfied
