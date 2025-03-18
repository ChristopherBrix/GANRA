import parse_smt2
import sys
import utils

query_intro = """
You are an expert at writing highly efficient PyTorch code.

Please write efficient Python code using PyTorch to solve the
following problem:
As an input, you receive a list of ‘K‘ formulas. Each formula is
given as a nested list that represents a tree. The first element
of inner nodes is the operation ("+", "-", "*" or "/") and which
is supposed to be applied to the following elements. The leafs
are strings or floating point numbers. Strings refer to variable
names. You also receive an input tensor ‘x‘ of shape ‘[batch_size,
N]‘, where ‘N‘ is the number of variables across all formulas.
Finally, you receive a list ‘inputs‘ that contains the names of
all variables across all formulas. The order of entries maps the
variable names to their indices in ‘x‘.
As an output, you should return a tensor of shape ‘[batch_size,
K]‘, with the corresponding results for each formula.

Important: Your computation must be highly efficient! The code
can utilize a GPU, so you should use this to parallelize it
as much as possible. You must not use for-loops for any tensor
related operation. Instead, use tensor operations to perform all
required operations in parallel. Do not rely on JIT compilation
for speedups, but detect patterns in the operations so you can use
matrix-matrix operations and other torch features to parallelize
them. If formulas depend on multiple copies of some sub-term ‘a‘,
do not recompute ‘a‘ for each instance. Instead, compute it only
once and then use broadcasting or advanced indexing to duplicate
the result.
Try to merge the operations in different branches of the tree: If
multiple branches perform the same type of operation, compute them
in parallel. E.g. instead of writing ‘a = x[:, 0]; b = x[:, 1];
a_squared = a ** 2; b_squared = b ** 2‘, write ‘squared = x[:, :2]
** 2; a_squared = squared[:, 0]; b_squared = squared[:, 1]‘. This
merges the squaring of ‘a‘ and ‘b‘.
Your code doesn’t have to parse arbitrary input formulas. Below,
you will see an example input. Extrapolate from this to detect
the underlying patterns. You do not have to account for any kind
of input that doesn’t exhibit the same patterns. However, your
code must not hard code any of these values, but be usable for any
other input with similar structure.
"""


query_outro = """
Perform the following steps: First, analyze the example and detect
the underlying patterns. Finally, implement the logic and provide
the python code. Your code must be able to deal with negative
variables, too. So do not use logarithms. Make sure to place all
tensors on the same device that the x tensor is on.
Write a function ‘compute_result(formulas, x, inputs)‘ that
receives the formulas, the variables x and the mapping to variable
names as an input, and returns a tensor with the corresponding
results.
"""


def main():
    path = sys.argv[1]
    script = parse_smt2.get_formula(path)
    print(query_intro + "\n" + str(utils.extract_formula(script)) + "\n" + query_outro)


if __name__ == "__main__":
    main()
