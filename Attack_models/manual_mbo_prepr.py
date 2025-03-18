from general_attack_model import *


class ManualMboPreprocessed(GeneralAttackModel):

    def __init__(self, path, script, epsilon, device, l2o_version):
        super().__init__(path, script, epsilon, device, l2o_version)
        self.max_power, sum_index, self.scalars = self.get_summands_of_mbo(
            path, device=device, is_preprocessed_mbo=True
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

        for var_index in self.sum_index:  # loop over variables (max: 7)
            indices = var_index.unsqueeze(0).expand(
                batch_size, -1
            )  # Shape: [batch_size, num_indices]
            factors = torch.gather(
                precomps, dim=1, index=indices
            )  # Shape: [batch_size, num_indices]
            res = torch.mul(res, factors)

        assert res.shape[1] % 2 == 0
        pos_res = res[:, : int(res.shape[1] / 2)]
        neg_res = res[:, int(res.shape[1] / 2) :]
        pos_summed = torch.sum(pos_res, dim=1)
        neg_summed = torch.sum(neg_res, dim=1)
        pos_is_positive = self._l2o_leq(-pos_summed)
        pos_is_positive_satisfied = -pos_summed <= 0
        neg_is_negative = self._l2o_leq(neg_summed)
        neg_is_negative_satisfied = neg_summed <= 0

        pos_constr = self._l2o_lt(-x)
        pos_constr_satisfied = (-x < 0).all(dim=1)

        conditions = torch.cat(
            [pos_is_positive.unsqueeze(1), neg_is_negative.unsqueeze(1), pos_constr],
            dim=1,
        )
        all_conditions_satisfied = (
            pos_is_positive_satisfied & neg_is_negative_satisfied & pos_constr_satisfied
        )
        return self._l2o_and(conditions), all_conditions_satisfied
