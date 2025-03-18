import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

TIMELIMIT = 600


def get_results(path):
    """
    Returns results for all files in the path

    Input: The path to the results.
    Output: A dictionary containing a list 'runtimes' (in ascending order), a dictionary list 'verified' that contains the runtime for each verified file and a dictionary 'unverified' that contains the runtime for each unverified file.
    """
    runtimes = []
    verified = dict()
    unverified = dict()
    for file in os.listdir(path):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(path, file), "r", errors="ignore") as f:
            runtime = -100
            lines = f.readlines()
            is_sat = False
            for line in lines:
                if not isinstance(line, str):
                    breakpoint()
                line = line.strip()
                if "sat" == line or "is valid" in line:
                    is_sat = True
                if line.startswith("real"):
                    runtime_str = line.split("\t")[1].strip()
                    runtime = float(runtime_str.split("m")[0]) * 60 + float(
                        runtime_str.split("m")[1][:-1]
                    )
            if runtime >= TIMELIMIT:
                assert not is_sat
                is_sat = False
                runtime = TIMELIMIT
            if is_sat:
                verified[file] = runtime
            else:
                runtime = TIMELIMIT
                unverified[file] = runtime
            runtime = min(runtime, TIMELIMIT)
            runtimes.append(runtime)
    runtimes.sort()
    return {
        "runtimes": runtimes,
        "verified": verified,
        "unverified": unverified,
    }


def load_data(experiments, compute_subset_over, highlight_best=True):
    results = {}
    for label, file in experiments:
        results[label] = get_results(file)

    if len(compute_subset_over) > 0:
        verified_subset = set(results[compute_subset_over[0]]["verified"].keys())
        for label in compute_subset_over[1:]:
            verified_subset = verified_subset.intersection(
                set(results[label]["verified"].keys())
            )
        for label, file in experiments:
            if label in compute_subset_over:
                results[label]["avg_runtime"] = sum(
                    [results[label]["verified"][file] for file in verified_subset]
                ) / len(verified_subset)
            else:
                results[label]["avg_runtime"] = "N/A"
        print(f"Intersection of verified instances: {len(verified_subset)}")

        lowest_runtime = min(
            [
                results[label]["avg_runtime"]
                for label, _ in experiments
                if label in compute_subset_over
            ]
        )
        for label, _ in experiments:
            if label in compute_subset_over:
                if highlight_best and results[label]["avg_runtime"] == lowest_runtime:
                    results[label][
                        "avg_runtime"
                    ] = f"\\textbf{{{results[label]['avg_runtime']:.2f}}}"
                else:
                    results[label][
                        "avg_runtime"
                    ] = f"{results[label]["avg_runtime"]:.2f}"
            else:
                results[label]["avg_runtime"] = "-"

    best_verification_count = max(
        [len(results[label]["verified"]) for label, _ in experiments]
    )
    for label, _ in experiments:
        if (
            highlight_best
            and len(results[label]["verified"]) == best_verification_count
        ):
            results[label][
                "verified_count"
            ] = f"\\textbf{{{len(results[label]['verified'])}}}"
        else:
            results[label]["verified_count"] = str(len(results[label]["verified"]))

    return results


def print_table(experiments, compute_subset_over):
    results = load_data(experiments, compute_subset_over)

    print(
        f"""
\\begin{{tabular}}{{@{{}}lllllllll@{{}}}} \\toprule
& Z3 & CVC5 & \\ugotNL/ & \\NRAgo/ & \\multicolumn{{4}}{{c}}{{\\oursolver/ (ours)}} \\\\ \\cmidrule(l){{6-9}}
& & & & & seq. & \\multicolumn{{2}}{{c}}{{manual}} & LLM \\\\ \\cmidrule(lr){{7-8}}
& & & & \\multicolumn{{1}}{{r}}{{$\\eps = $}} & $10^{{-4}}$ & $0$ & $10^{{-4}}$ & $10^{{-4}}$ \\\\ \\midrule
SAT [$\\uparrow$] & {' & '.join([results[tool]['verified_count'] for tool, _ in experiments])} \\\\
avg. runtime [s] [$\\downarrow$] & {' & '.join([results[tool]['avg_runtime'] for tool, _ in experiments])} \\\\ \\bottomrule
\\end{{tabular}}
"""
    )


def print_table_mbo_preprocess(experiments):
    results = load_data(experiments, [], highlight_best=False)

    print(
        f"""
\\begin{{tabular}}{{@{{}}lllllllll@{{}}}} \\toprule
& \\multicolumn{{2}}{{c}}{{Z3}} & \\multicolumn{{2}}{{c}}{{CVC5}} & \\multicolumn{{2}}{{c}}{{\\ugotNL/}} & \\multicolumn{{2}}{{c}}{{\\NRAgo/}} \\\\ \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}} \\cmidrule(l){{8-9}}
& Orig & Interval & Orig & Interval & Orig & Interval & Orig & Interval \\\\ \\midrule
SAT [$\\uparrow$] & {' & '.join([results[tool]['verified_count'] for tool, _ in experiments])} \\\\ \\bottomrule
\\end{{tabular}}
"""
    )


def print_table_l2o(
    kissing_experiments,
    kissing_compute_subset_over,
    mbo_experiments,
    mbo_compute_subset_over,
):
    kissing_results = load_data(kissing_experiments, kissing_compute_subset_over)
    mbo_results = load_data(mbo_experiments, mbo_compute_subset_over)

    results = kissing_results | mbo_results
    experiments = kissing_experiments + mbo_experiments

    print(
        f"""
\\begin{{tabular}}{{@{{}}lllllll@{{}}}} \\toprule
& \\multicolumn{{3}}{{c}}{{Kissing}} & \\multicolumn{{3}}{{c}}{{Sturm-MBO}} \\\\ \\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
L2O version & \\oursolver/ & \\NRAgo/ & \\ugotNL/ & \\oursolver/ & \\NRAgo/ & \\ugotNL/ \\\\ \\midrule
SAT [$\\uparrow$] & {' & '.join([results[tool]['verified_count'] for tool, _ in experiments])} \\\\
avg. runtime [s] [$\\downarrow$] & {' & '.join([results[tool]['avg_runtime'] for tool, _ in experiments])} \\\\ \\bottomrule
\\end{{tabular}}
"""
    )


# Kissing table
if False:  # TODO: check if result verification works, edited paths
    experiments = [
        ("Z3", "results/kissing_z3"),
        ("CVC5", "results/kissing_cvc5"),
        ("UGOT", "results/kissing_ugotnl"),
        ("NRAgo", "results/kissing_nrago"),
        ("ganra_seq", "results/kissing_attacker/forward_logic_sequential"),
        (
            "ganra_man0",
            "results/kissing_attacker/forward_logic_manual_kissing__epsilon_0",
        ),
        ("ganra_maneps", "results/kissing_attacker/forward_logic_manual_kissing"),
        ("ganra_gpt", "results/kissing_attacker/forward_logic_gpt_kissing"),
    ]
    compute_subset_over = ["Z3", "NRAgo", "ganra_seq", "ganra_maneps", "ganra_gpt"]
    print_table(experiments, compute_subset_over)

# L2O table
if False:
    kissing_experiments = [
        (
            "kissing_ganra_maneps",
            "results/kissing_attacker/forward_logic_manual_kissing",
        ),
        (
            "kissing_nrago",
            "results/kissing_attacker/forward_logic_manual_kissing__l2o_version_nrago",
        ),
        (
            "kissing_ugotnl",
            "results/kissing_attacker/forward_logic_manual_kissing__l2o_version_ugotnl",
        ),
    ]
    kissing_compute_subset_over = [
        "kissing_ganra_maneps",
        "kissing_nrago",
        "kissing_ugotnl",
    ]
    mbo_experiments = [
        (
            "mbo_ganra_maneps",
            "results/mbo_preprocessed_attacker/seed_0__step_size_1e-3__attack_adam__l2o_version_ours__forward_logic_manual_mbo_preprocessed__batch_size_10000__steps_1000",
        ),
        (
            "mbo_nrago",
            "results/mbo_preprocessed_attacker/l2o_version_nrago__forward_logic_manual_mbo_preprocessed",
        ),
        (
            "mbo_ugotnl",
            "results/mbo_preprocessed_attacker/l2o_version_ugotnl__forward_logic_manual_mbo_preprocessed",
        ),
    ]
    mbo_compute_subset_over = ["mbo_ganra_maneps", "mbo_nrago", "mbo_ugotnl"]
    print_table_l2o(
        kissing_experiments,
        kissing_compute_subset_over,
        mbo_experiments,
        mbo_compute_subset_over,
    )


# MBO table
if False:
    experiments = [
        ("Z3", "results/mbo_orig_z3"),
        ("CVC5", "results/mbo_orig_cvc5"),
        ("UGOT", "results/mbo_orig_ugotnl"),
        ("NRAgo", "results/mbo_orig_nrago"),
        (
            "ganra_seq",
            "results/mbo_preprocessed_attacker/seed_0__step_size_1e-3__attack_adam__l2o_version_ours__forward_logic_sequential__batch_size_10000__steps_1000",
        ),
        (
            "ganra_man0",
            "results/mbo_preprocessed_attacker/seed_0__step_size_1e-3__attack_adam__l2o_version_ours__forward_logic_manual_mbo_preprocessed__batch_size_10000__steps_1000__epsilon_0",
        ),
        (
            "ganra_maneps",
            "results/mbo_preprocessed_attacker/seed_0__step_size_1e-3__attack_adam__l2o_version_ours__forward_logic_manual_mbo_preprocessed__batch_size_10000__steps_1000",
        ),
        (
            "ganra_gpt",
            "results/mbo_preprocessed_attacker/seed_0__step_size_1e-3__attack_adam__l2o_version_ours__forward_logic_gpt_mbo_preprocessed__batch_size_10000__steps_1000",
        ),
    ]
    compute_subset_over = [
        "UGOT",
        "ganra_seq",
        "ganra_man0",
        "ganra_maneps",
        "ganra_gpt",
    ]
    print_table(experiments, compute_subset_over)

# MBO orig vs. preprocessed table
if False:
    experiments = [
        ("Z3_orig", "results/mbo_orig_z3"),
        ("Z3_preprocessed", "results/mbo_preprocessed_z3"),
        ("CVC5_orig", "results/mbo_orig_cvc5"),
        ("CVC5_preprocessed", "results/mbo_preprocessed_cvc5"),
        ("UGOT_orig", "results/mbo_orig_ugotnl"),
        ("UGOT_preprocessed", "results/mbo_preprocessed_ugotnl"),
        ("NRAgo_orig", "results/mbo_orig_nrago"),
        ("NRAgo_preprocessed", "results/mbo_preprocessed_nrago"),
    ]
    print_table_mbo_preprocess(experiments)

import create_mbo_benchmarks


def _is_sat(filename):
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "sat":
                return True
    return False


verified_z3 = []
verified_attack = []
existing = []
for summands in range(25, 99, 25):
    verified_z3.append([])
    verified_attack.append([])
    existing.append([])
    for N in range(1, 31, 1):
        verified_z3[-1].append(0)
        verified_attack[-1].append(0)
        existing[-1].append(0)
        for i in range(10):
            # test with create_mbo_benchmarks load_result
            is_sat_instance = create_mbo_benchmarks.load_result(
                f"mbo_{summands}_{N}_{i}"
            )
            if is_sat_instance != "sat":
                continue

            z3_not_found = False
            try:
                verified_z3[-1][-1] += _is_sat(
                    f"results/custom_mbo/fst_mbo_{summands}_{N}_{i}_z3.txt"
                )
            except FileNotFoundError:
                z3_not_found = True

            attack_not_found = False
            try:
                verified_attack[-1][-1] += _is_sat(
                    f"results/custom_mbo/fst_mbo_{summands}_{N}_{i}_attack.txt"
                )
            except FileNotFoundError:
                attack_not_found = True

            if z3_not_found or attack_not_found:
                if not z3_not_found and not attack_not_found:
                    print("Warning", (summands, N, i), z3_not_found, attack_not_found)
            else:
                existing[-1][-1] += 1
print(f"existing = {existing}")
print(f"verified_z3 = {verified_z3}")
print(f"verified_attack = {verified_attack}")


# # For all files in path
# folder = "results/kissing_attacker/"
# for label, file in [
#     ("Z3", "/results_bkp/kissing_z3"),
#     ("Attacker sz1e-3 adam", folder + "bs10000_st1000_sz1e-3_s0_attackadam"),
#     ("Attacker sz1e-4 adam", folder + "bs10000_st1000_sz1e-4_s0_attackadam"),
#     ("Attacker sz1e-4 signsgd seed 0", folder + "bs10000_st1000_sz1e-4_s0_attacksignsgd"),
#     ("Attacker sz1e-4 signsgd seed 1", folder + "bs10000_st1000_sz1e-4_s1_attacksignsgd"),
#     ("Attacker sz1e-3 adam v2", folder + "bs10000_st1000_sz1e-3_s0_attackadam_versionv2"),
#     ("Attacker sz1e-3 adam v3", folder + "bs10000_st1000_sz1e-3_s0_attackadam_versionv3"),
#     ("Attacker sz1e-3 adam v5", folder + "bs10000_st1000_sz1e-3_s0_attackadam_versionv5"),
#     ("Ours", folder + "bs10000_st1000_sz1e-3_s0_attackadam_l2oours")
# ]:
#     runtimes, verified, unverified = get_results(file)
#     plt.plot(runtimes, label=label)
#     print(f"{label}: unverified files: {unverified}")

# plt.xlabel("Instances")
# plt.ylabel("Runtime (s)")
# plt.title("Z3 runtimes")
# plt.legend()
# plt.show()
