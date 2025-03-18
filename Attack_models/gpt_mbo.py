import torch
from general_attack_model import *


class GPTMbo(GeneralAttackModel):

    def __init__(self, path, script, epsilon, device, l2o_version):
        super().__init__(path, script, epsilon, device, l2o_version)
        self.smt_form = utils.extract_formula(script)

    def forward(self, x):
        gpt_res = compute_result(
            [self.smt_form[1][7][1], self.smt_form[2][7][1]], x, self.inputs
        )
        pos_summed, neg_summed = gpt_res[:, 0], gpt_res[:, 1]

        pos_is_positive = self._l2o_leq(pos_summed)
        pos_is_positive_satisfied = pos_summed <= 0
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


def compute_result(formulas, x, inputs):
    device = x.device
    batch_size = x.size(0)
    num_vars = len(inputs)

    # Map variable names to indices in x
    varname_to_index = {var_name: idx for idx, var_name in enumerate(inputs)}

    terms_coefficients = []
    terms_exponents = []
    terms_formula_idx = []

    # Function to parse nodes recursively and collect terms
    def parse_node(node):
        if isinstance(node, float):
            return [(node, {})]
        elif isinstance(node, str):
            return [(1.0, {node: 1})]
        elif isinstance(node, list):
            op = node[0]
            operands = node[1:]
            if op == "+":
                terms = []
                for operand in operands:
                    terms.extend(parse_node(operand))
                return terms
            elif op == "-":
                if len(operands) == 1:
                    # Unary minus
                    terms = parse_node(operands[0])
                    return [(-coeff, exponents) for coeff, exponents in terms]
                elif len(operands) == 2:
                    # Binary subtraction
                    terms1 = parse_node(operands[0])
                    terms2 = parse_node(operands[1])
                    negated_terms2 = [
                        (-coeff, exponents) for coeff, exponents in terms2
                    ]
                    return terms1 + negated_terms2
                else:
                    raise ValueError(
                        f"Unsupported '-' operation with operands: {operands}"
                    )
            elif op == "*":
                # Compute product of terms
                terms = [(1.0, {})]
                for operand in operands:
                    operand_terms = parse_node(operand)
                    new_terms = []
                    for coeff1, exponents1 in terms:
                        for coeff2, exponents2 in operand_terms:
                            coeff = coeff1 * coeff2
                            exponents = exponents1.copy()
                            for var, exp in exponents2.items():
                                exponents[var] = exponents.get(var, 0) + exp
                            new_terms.append((coeff, exponents))
                    terms = new_terms
                return terms
            else:
                raise ValueError(f"Unsupported operation '{op}'")
        else:
            raise ValueError(f"Unsupported node type: {node}")

    # Parse each formula
    for formula_idx, formula in enumerate(formulas):
        terms = parse_node(formula)
        for coeff, exponents in terms:
            terms_coefficients.append(coeff)
            terms_exponents.append(exponents)
            terms_formula_idx.append(formula_idx)

    num_terms = len(terms_coefficients)

    # Prepare exponents tensor
    exponents_tensor = torch.zeros(
        (num_terms, num_vars), dtype=torch.float32, device=device
    )
    for term_idx, exponents in enumerate(terms_exponents):
        for var_name, exponent in exponents.items():
            var_idx = varname_to_index[var_name]
            exponents_tensor[term_idx, var_idx] = exponent

    # Prepare coefficients tensor
    coefficients_tensor = torch.tensor(
        terms_coefficients, dtype=torch.float32, device=device
    )

    # Prepare term formula indices tensor
    terms_formula_idx_tensor = torch.tensor(
        terms_formula_idx, dtype=torch.long, device=device
    )

    # Compute x to the power of exponents
    x_expanded = x.unsqueeze(1)  # Shape: [batch_size, 1, num_vars]
    exponents_expanded = exponents_tensor.unsqueeze(
        0
    )  # Shape: [1, num_terms, num_vars]
    x_powers = torch.pow(
        x_expanded, exponents_expanded
    )  # Shape: [batch_size, num_terms, num_vars]

    # Compute term values
    term_products = torch.prod(x_powers, dim=2)  # Shape: [batch_size, num_terms]
    term_values = (
        coefficients_tensor.unsqueeze(0) * term_products
    )  # Shape: [batch_size, num_terms]

    # Compute outputs
    num_formulas = len(formulas)
    outputs = torch.zeros(
        (batch_size, num_formulas), dtype=torch.float32, device=device
    )
    outputs.scatter_add_(
        1,
        terms_formula_idx_tensor.unsqueeze(0).expand(batch_size, num_terms),
        term_values,
    )

    return outputs
