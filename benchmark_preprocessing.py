import os
import re


def preprocessing_mbo(filename, variables):
    """
    Creates a modified version of the mbo file (given as filename) where each variable is replaced by two variables, one ensuring that there exists a solution such that the sum is smaller than 0, the other ensuring a solution greater than 0.

    Input: The path to the given mbo file and a list of variables occuring in it.
    Output: The path to the newly generated file.
    """
    base_path = "preprocessed_mbo/"
    base_name, ext = os.path.splitext(filename)
    filepath = os.path.join(base_path, filename)
    counter = 1

    while os.path.exists(filepath):
        filepath = os.path.join(base_path, f"{base_name}_{counter}{ext}")
        counter += 1

    with open(filepath, "w") as f:
        with open(f"{filename}", "r") as orig_file:
            orig_lines = orig_file.readlines()
        while len(orig_lines) > 0:
            if orig_lines[0].startswith("(declare-const"):
                break
            f.write(orig_lines.pop(0))
        f.write("\n")
        for var in variables:
            # add variables to ensure Formula > 0 and Formula < 0 are satisfiable
            f.write(f"(declare-const {var}p Real)\n")
            f.write(f"(declare-const {var}n Real)\n")
        f.write("\n")

        while len(orig_lines) > 0 and "(assert" not in orig_lines[0]:
            orig_lines.pop(0)

        buffer = ""
        while len(orig_lines) > 0 and "check-sat" not in orig_lines[0]:
            buffer += orig_lines.pop(0)

        pos_assertions = buffer
        neg_assertions = buffer

        pos_assertions = pos_assertions.replace("=", ">")
        neg_assertions = neg_assertions.replace("=", "<")

        # integrate all assertions into one assert statement
        pos_assertions = re.sub(f'{re.escape("))")}(?!.*{re.escape("))")})', '', pos_assertions, 1) 
        neg_assertions = re.sub(re.escape("(assert (and"), '', neg_assertions, 1)

        for var in variables:
            pos_assertions = pos_assertions.replace(var, var + "p")
            neg_assertions = neg_assertions.replace(var, var + "n")

        f.write(pos_assertions)
        f.write(neg_assertions)
        while len(orig_lines) > 0:
            f.write(orig_lines.pop(0))
    return filepath
