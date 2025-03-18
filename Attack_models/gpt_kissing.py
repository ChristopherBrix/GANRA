from general_attack_model import *


class GPTKissing(GeneralAttackModel):

    def __init__(self, path, script, epsilon, device, l2o_version):
        super().__init__(path, script, epsilon, device, l2o_version)
        self.smt_form = utils.extract_formula(script)
        self.equality_indices = []
        self.inequality_indices = []
        if self.smt_form[0] == "and":
            formulas = self.smt_form[1:]
        else:
            formulas = [self.smt_form]
        for i, formula in enumerate(formulas):
            if formula[0] == "=":
                self.equality_indices.append(i)
            elif formula[0] == "<=":
                self.inequality_indices.append(i)
            else:
                raise ValueError(f"Expected only = and <=", formula[0])

    def forward(self, x):
        if self.smt_form[0] == "and":
            formulas = self.smt_form[1:]
        else:
            formulas = [self.smt_form]
        equations = [formulas[i][1] for i in range(0, len(formulas))]
        results = compute_result(equations, x, self.inputs)
        equalities = results[:, self.equality_indices]
        inequalities = results[:, self.inequality_indices]
        conditions = torch.cat(
            [self._l2o_eq(equalities), self._l2o_leq(inequalities)], dim=1
        )
        all_conditions_satisfied = (equalities.abs() <= self.epsilon).all(dim=1) & (
            inequalities <= 0
        ).all(dim=1)
        return self._l2o_and(conditions), all_conditions_satisfied


def compute_result(formulas, x, inputs):
    device = x.device
    batch_size = x.shape[0]

    # Build mapping from variable names to indices in x
    variable_mapping = {}
    for idx, var_name in enumerate(inputs):
        tokens = var_name.split("_")
        if len(tokens) != 3 or not tokens[1].isdigit() or not tokens[2].isdigit():
            continue  # Skip invalid variable names
        i = int(tokens[1])
        k = int(tokens[2])
        variable_mapping[var_name] = {"i": i, "k": k, "index_in_x": idx}

    # Map i to list of (k, idx_in_x)
    i_to_k_idx = {}
    for var_name, data in variable_mapping.items():
        i = data["i"]
        k = data["k"]
        idx_in_x = data["index_in_x"]
        if i not in i_to_k_idx:
            i_to_k_idx[i] = []
        i_to_k_idx[i].append((k, idx_in_x))

    # Get sorted list of unique i's
    unique_is = sorted(i_to_k_idx.keys())
    num_i = len(unique_is)

    # Build x_i_list
    x_i_list = []
    for i in unique_is:
        k_idx_pairs = sorted(i_to_k_idx[i], key=lambda x: x[0])  # Sort by k
        idxs_for_i = [idx_in_x for k, idx_in_x in k_idx_pairs]
        x_i = x[:, idxs_for_i]  # Shape [batch_size, D]
        x_i_list.append(x_i)

    # Stack x_i tensors
    X = torch.stack(x_i_list, dim=0).to(device)  # Shape [num_i, batch_size, D]

    # Transpose to shape [batch_size, num_i, D]
    X = X.permute(1, 0, 2)  # Shape [batch_size, num_i, D]

    # Compute x_i squared norms
    x_i_squared_norms = (X**2).sum(dim=2)  # Shape [batch_size, num_i]

    s_i = x_i_squared_norms - 1.0  # Shape [batch_size, num_i]

    # Compute pairwise squared distances
    x_i_xj_dot = torch.bmm(X, X.transpose(1, 2))  # Shape [batch_size, num_i, num_i]

    # Compute s_ij = 1.0 - ||x_i - x_j||^2
    s_ij = 1.0 - (
        x_i_squared_norms.unsqueeze(2) + x_i_squared_norms.unsqueeze(1) - 2 * x_i_xj_dot
    )  # Shape [batch_size, num_i, num_i]

    # Extract upper triangle indices
    indices_i, indices_j = torch.triu_indices(num_i, num_i, offset=1)

    # s_ij_upper has shape [batch_size, num_pairs]
    s_ij_upper = s_ij[:, indices_i, indices_j]

    # Concatenate s_i and s_ij_upper
    output = torch.cat([s_i, s_ij_upper], dim=1)  # Shape [batch_size, K]

    return output
