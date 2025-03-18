import os
import random
import math
import sys
import benchmark_preprocessing
import create_my_model
from array_jobs import experiments

general_smt_header = """ (set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :series |mbo-like problems|)
(set-info :category "industrial")
(set-info :status unknown)
"""


def create_benchmark(
    summand_factor_limit,
    pathname,
    max_of_summands=None,
    absolute_num_summands=None,
    scalar_maximum=20,
):
    # include j2 in 40% of the cases
    with_j2 = random.random() >= 0.4
    if with_j2:
        variables = ["h1", "h2", "h3", "h4", "h5", "h6", "j2"]
    else:
        variables = ["h1", "h2", "h3", "h4", "h5", "h6"]

    if absolute_num_summands == None:
        summands_num = math.floor(random.betavariate(alpha=8, beta=2) * max_of_summands)
    else:
        summands_num = absolute_num_summands

    declarations = ""

    for var in variables:
        declarations += f"(declare-const {var} Real)\n"

    pos_constraints = "(assert (and"
    for var in variables:
        pos_constraints += f" (> {var} 0)"
    pos_constraints

    summands = []

    for _ in range(summands_num):
        cuts = sorted(random.choices(range(0, summand_factor_limit + 1), k=5))
        power_of_vars = (
            [cuts[0]]
            + [cuts[i] - cuts[i - 1] for i in range(1, 5)]
            + [summand_factor_limit - cuts[-1]]
        )

        if with_j2:
            # j2 can have an arbitrary power from 0 up to the limit
            power = random.randint(0, summand_factor_limit)
            power_of_vars.append(power)

        negative = random.random() >= 0.8
        scalar = random.randint(1, scalar_maximum)

        if negative:
            tmp = f"(* -{scalar}"
        else:
            tmp = f"(* {scalar}"

        for ind, power in enumerate(power_of_vars):
            var = variables[ind]
            if power == 1:
                tmp += " " + var
            elif power == 0:
                pass
            else:
                tmp += " (*"
                for _ in range(power):
                    tmp += " " + var
                tmp += ")"
        tmp += ")"
        summands.append(tmp)

    equality_constraint = f"""(= (+ {" ".join(map(str, summands))}) 0))) 
(check-sat)
(exit)"""

    with open(pathname, "w") as f:
        f.write(general_smt_header)
        f.write(
            f"(set-info :source | summand factor limit is {summand_factor_limit}, mean of summands is {max_of_summands}|)\n"
        )
        f.write(declarations)
        f.write(pos_constraints)
        f.write(equality_constraint)


def load_result(result_filename):
    result = "unknown"
    with open(f"self_generated_mbo/results/fst_{result_filename}.txt") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "sat":
                assert result == "unknown"
                result = "sat"
            elif line == "unsat":
                assert result == "unknown"
                result = "unsat"
    return result


def test_z3(result_filename, pathname):
    z3_command = experiments.get_z3_command(
        result_filename, pathname, "self_generated_mbo/results", timeout=180
    )
    os.system(z3_command)
    result = load_result(result_filename)
    return result


def test_attacker(result_filename, pathname_preprocessed):
    attacker_command = experiments.get_attacker_command(
        None,
        result_filename,
        pathname_preprocessed,
        {"l2o_version": "ours", "forward_logic": "manual_mbo_preprocessed"},
        timeout=180,
        result_dir="self_generated_mbo/results/",
    )
    os.system(attacker_command)
    result = load_result(result_filename)
    return result


def main():  # called when generate_custom mbo is enabled
    if len(sys.argv) < 4:
        print("Usage: python create_mbo_benchmarks.py <num_summands> <N> <i>")
        sys.exit(1)

    num_summands = int(sys.argv[1])
    N = int(sys.argv[2])
    i = int(sys.argv[3])

    while True:
        pathname = f"self_generated_mbo/mbo_{num_summands}_{N}_{i}.smt2"
        create_benchmark(
            summand_factor_limit=N,
            pathname=pathname,
            scalar_maximum=20,
            absolute_num_summands=num_summands,
        )
        model, _, _ = create_my_model.create_my_model(
            pathname,
            epsilon=0.0001,
            device="cpu",
            l2o_version="ours",
            forward_logic="manual_mbo_orig",
        )
        variables = model.variables
        pathname_preprocessed = benchmark_preprocessing.preprocessing_mbo(
            pathname, variables=variables
        )

        result_filename = f"mbo_{num_summands}_{N}_{i}"

        attacker_res = test_attacker(result_filename, pathname)
        if attacker_res != "sat":
            print(
                "Could not solve the benchmark with our solver, trying Z3", flush=True
            )
            z3_res = test_z3(result_filename, pathname_preprocessed)
            if z3_res == "sat":
                print("Z3 found a solution", flush=True)
                break
            else:
                print(f"Z3 did not find a solution ({z3_res})", flush=True)
        elif attacker_res == "sat":
            print("Our solver found a solution", flush=True)
            break

        print("Could not solve the benchmark, trying again", flush=True)
        os.remove(pathname)
        os.remove(pathname_preprocessed)
        os.remove(f"self_generated_mbo/results/fst_{result_filename}.txt")


if __name__ == "__main__":
    main()
