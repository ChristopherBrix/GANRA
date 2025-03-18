import copy


def get_variables(script):
    form = script.get_last_formula()
    return sorted([str(x) for x in form.get_free_variables()])


def extract_formula(script):
    form = script.get_last_formula()
    smt_form = _formula_adapter(form)
    smt_form = _formula_simplifier(smt_form)

    return smt_form


def _extract_input(script):
    form = script.get_last_formula()
    inputs = sorted([str(x) for x in form.get_free_variables()])
    return form, inputs


def _formula_adapter(formula):
    if formula.is_and():
        assert len(formula.args()) > 0, "No child nodes. "
        children = [_formula_adapter(arg) for arg in formula.args()]
        return ["and"] + children
    elif formula.is_or():
        assert len(formula.args()) > 0, "No child nodes. "
        children = [_formula_adapter(arg) for arg in formula.args()]
        return ["or"] + children
    elif formula.is_not():
        return ["not", _formula_adapter(formula.arg(0))]
    elif formula.is_le():
        return ["<=", _adapter_helper(formula.arg(0)), _adapter_helper(formula.arg(1))]
    elif formula.is_lt():
        return ["<", _adapter_helper(formula.arg(0)), _adapter_helper(formula.arg(1))]
    elif formula.is_equals():
        return ["=", _adapter_helper(formula.arg(0)), _adapter_helper(formula.arg(1))]
    else:
        print(f"Not supported for type {formula.node_type()}")
        raise TypeError


def _formula_simplifier(formula):
    """
    Applies simplifications to the formula until it becomes stable.
    """

    made_changes = True
    while made_changes:
        old_formula = copy.deepcopy(formula)
        formula = _formula_simplifier_remove_not(formula)
        formula = _formula_simplifier_remove_gt(formula)
        made_changes = formula != old_formula
    formula = _formula_simplifier_compare_to_0(formula)
    return formula


def _formula_simplifier_remove_gt(formula):
    """
    Removes > by inverting it.

    Input: The output of _formula_adapter, i.e. a list where the first element is the operation and the rest
    are the arguments.
    Output: A list with the same structure, but with > reverted.
    """
    if isinstance(formula, list):
        if formula[0] == ">":
            assert len(formula) == 3, "Wrong number of arguments for >. "
            return ["<="] + [
                _formula_simplifier_remove_gt(formula[2]),
                _formula_simplifier_remove_gt(formula[1]),
            ]
        elif formula[0] == ">=":
            assert len(formula) == 3, "Wrong number of arguments for >=."
            return ["<"] + [
                _formula_simplifier_remove_gt(formula[2]),
                _formula_simplifier_remove_gt(formula[1]),
            ]
        else:
            return [formula[0]] + [
                _formula_simplifier_remove_gt(arg) for arg in formula[1:]
            ]
    else:
        return formula


def _formula_simplifier_compare_to_0(formula):
    """
    Converts a X b to a - b X 0.

    Input: The output of _formula_adapter, i.e. a list where the first element is the operation and the rest
    are the arguments.
    Output: A list with the same structure, but with a X b converted to a X b = 0.
    """
    if isinstance(formula, list):
        assert formula[0] not in [
            ">=",
            ">",
        ], ">= and > should have been removed by now."
        if formula[0] in ["<", "<=", "="]:
            assert len(formula) == 3, "Wrong number of arguments for comparison."
            if formula[2] == 0:
                return formula
            elif formula[1] == 0 and formula[0] == "=":
                return [formula[0], formula[2], 0]
            return [formula[0], ["-", formula[1], formula[2]], 0]
        else:
            return [formula[0]] + [
                _formula_simplifier_compare_to_0(arg) for arg in formula[1:]
            ]
    else:
        return formula


def _formula_simplifier_remove_not(formula):
    """
    Removes NOT by pushing it down the tree.

    Input: The output of _formula_adapter, i.e. a list where the first element is the operation and the rest
    are the arguments.
    Output: A list with the same structure, but with NOT pushed down the tree.
    """
    if isinstance(formula, list):
        if formula[0] == "not":
            assert formula[1][0] != "=", "'not =' is not supported."
            if formula[1][0] == "not":
                return _formula_simplifier_remove_not(formula[1][1])
            elif formula[1][0] == "and":
                return ["or"] + [
                    _formula_simplifier_remove_not(["not"] + [arg])
                    for arg in formula[1][1:]
                ]
            elif formula[1][0] == "or":
                return ["and"] + [
                    _formula_simplifier_remove_not(["not"] + [arg])
                    for arg in formula[1][1:]
                ]
            elif formula[1][0] == "<":
                return [">="] + [
                    _formula_simplifier_remove_not(arg) for arg in formula[1][1:]
                ]
            elif formula[1][0] == "<=":
                return [">"] + [
                    _formula_simplifier_remove_not(arg) for arg in formula[1][1:]
                ]
            else:
                print(f"Not supported for type {formula[1][0]}")
                raise TypeError
        else:
            return [formula[0]] + [
                _formula_simplifier_remove_not(arg) for arg in formula[1:]
            ]
    else:
        return formula


def _adapter_helper(formula):
    if formula.is_plus():
        children = [_adapter_helper(arg) for arg in formula.args()]
        return ["+"] + children
    elif formula.is_minus():
        return ["-", _adapter_helper(formula.arg(0)), _adapter_helper(formula.arg(1))]
    elif formula.is_times():
        children = [_adapter_helper(arg) for arg in formula.args()]
        return ["*"] + children
    elif formula.is_div():
        return ["/", _adapter_helper(formula.arg(0)), _adapter_helper(formula.arg(1))]
    else:
        if formula.is_constant():
            res = float(formula.constant_value())
        else:  # input node
            res = str(formula)
        return res
