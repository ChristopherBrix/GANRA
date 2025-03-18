import argparse

experiments = {
    # "z3": ["kissing", "mbo_orig", "mbo_preprocessed"],
    # "cvc5": ["kissing", "mbo_orig", "mbo_preprocessed"],
    # "ugotnl": ["kissing", "mbo_orig", "mbo_preprocessed"],
    # "nrago": ["kissing", "mbo_orig", "mbo_preprocessed"],
    "AttackModel": {
        "kissing": [
            # {"forward_logic": "manual_kissing"},
            # {"forward_logic": "manual_kissing", "epsilon": 0},
            # {"forward_logic": "gpt_kissing"},
            # {"forward_logic": "sequential"},
            # {"forward_logic": "manual_kissing", "l2o_version": "ugotnl"},
            # {"forward_logic": "manual_kissing", "l2o_version": "nrago"},
            # {"forward_logic": "manual_kissing", "attack": "sgd"},
            # {"forward_logic": "manual_kissing", "attack": "signsgd"},
            # {"forward_logic": "manual_kissing", "attack": "sgd", "step_size": "1e-4"},
            # {"forward_logic": "manual_kissing", "attack": "signsgd", "step_size": "1e-4"},
            # {"forward_logic": "manual_kissing", "attack": "sgd", "step_size": "1e-2"},
            # {"forward_logic": "manual_kissing", "attack": "signsgd", "step_size": "1e-2"},
        ],
        "mbo_preprocessed": [
            # {"forward_logic": "manual_mbo_preprocessed"},
            # {"forward_logic": "manual_mbo_preprocessed", "seed": 1},
            # {"seed": 0, "step_size": "1e-3", "attack": "adam", "l2o_version": "ours", "forward_logic": "gpt_mbo_preprocessed", "batch_size": 10000, "steps": 1000},
            # {"seed": 0, "step_size": "1e-3", "attack": "adam", "l2o_version": "ours", "forward_logic": "manual_mbo_preprocessed", "batch_size": 10000, "steps": 1000, "epsilon": 0},
            # {"seed": 0, "step_size": "1e-3", "attack": "adam", "l2o_version": "ours", "forward_logic": "sequential", "batch_size": 10000, "steps": 1000},
            # {"forward_logic": "manual_mbo_preprocessed", "l2o_version": "ugotnl"},
            # {"forward_logic": "manual_mbo_preprocessed", "l2o_version": "nrago"},
            # {"forward_logic": "manual_mbo_preprocessed", "attack": "sgd"},
            # {"forward_logic": "manual_mbo_preprocessed", "attack": "signsgd"},
            # {"forward_logic": "manual_mbo_preprocessed", "attack": "sgd", "step_size": "1e-4"},
            # {"forward_logic": "manual_mbo_preprocessed", "attack": "signsgd", "step_size": "1e-4"},
            # {"forward_logic": "manual_mbo_preprocessed", "attack": "sgd", "step_size": "1e-2"},
            # {"forward_logic": "manual_mbo_preprocessed", "attack": "signsgd", "step_size": "1e-2"},
        ],
    }
}


def get_attacker_command(benchmark, i, file, config, timeout=600, result_dir=None):
    assert "sat_instance_output_dir" not in config
    if result_dir is None:
        result_dir = f"results/{benchmark}_AttackModel/{'__'.join([f'{k}_{v}' for k, v in config.items()])}"
    return f"""
source ../venv/bin/activate
module load GCCcore/.13.2.0 Z3/4.13.0
mkdir -p {result_dir}
{{ echo "{file}"; time timeout {timeout} python3 attacker.py {' '.join([f'--{k} {v}' for k, v in config.items()])} --sat_instance_output_file {result_dir}/sat_instances/found_sat_{i}.smt2 {file} ; }} > {result_dir}/fst_{i}.txt 2>&1
""".strip().replace(
        "\n", "; "
    )


def get_z3_command(i, file, result_dir, timeout=600):
    return f"""
mkdir -p {result_dir}
module load GCCcore/.13.2.0 Z3/4.13.0
{{ echo "{file}"; time timeout {timeout} z3 {file} ; }} > {result_dir}/fst_{i}.txt 2>&1
""".strip().replace(
        "\n", "; "
    )


def get_cvc5_command(i, file, result_dir, timeout=600):
    return f"""
mkdir -p {result_dir}
{{ echo "{file}"; time timeout {timeout} ../cvc5/bin/cvc5 {file} ; }} > {result_dir}/fst_{i}.txt 2>&1
""".strip().replace(
        "\n", "; "
    )


def get_ugot_command(
    i, file, result_dir, timeout=600
):  
    return f"""
source ../venv_ugotNL/bin/activate
module load GCCcore/.12.2.0 Python/3.10.8 Z3/4.12.2-Python-3.10.8
mkdir -p {result_dir}
{{ echo "{file}"; time timeout {timeout} python ../ugotNL/src/ugotNL.py {file} --epsilon_inflation --filter_overconstrV  --recursionlimit 10000 ; }} > {result_dir}/fst_{i}.txt 2>&1
""".strip().replace(
        "\n", "; "
    )


def get_nrago_command(i, file, result_dir, timeout=600):
    return f"""
source ../venv/bin/activate
mkdir -p {result_dir}
{{ echo "{file}"; time timeout {timeout} python NRAgo/nrago.py {file} ; }} > {result_dir}/fst_{i}.txt 2>&1
""".strip().replace(
        "\n", "; "
    )


def get_mbo_generation_command(num_summands, N, i):
    return f"""
source ../venv/bin/activate
mkdir -p self_generated_mbo
mkdir -p preprocessed_mbo
mkdir -p preprocessed_mbo/self_generated_mbo
python create_mbo_benchmarks.py {num_summands} {N} {i}
""".strip().replace(
        "\n", "; "
    )


def _get_files(benchmark):
    with open(f"random_files_{benchmark}.txt") as f:
        files = f.readlines()
        for file in files:
            file = file.strip()
            if file == "":
                continue
            yield file


def main():
    parser = argparse.ArgumentParser(
        description="Specify experiments to run to get according commands."
    )
    parser.add_argument(
        "--generate_custom_mbo",
        action="store_true",
        help="If custom mbo benchmark will be generated.",
    )
    parser.add_argument(
        "--evaluate_custom_mbo",
        action="store_true",
        help="If evaluation will take place on the custom mbo benchmark.",
    )

    args = parser.parse_args()
    generate_custom_mbo = args.generate_custom_mbo
    evaluate_custom_mbo = args.evaluate_custom_mbo

    for benchmark in experiments.get("z3", []):
        result_dir = f"results/{benchmark}_z3"
        for i, file in enumerate(_get_files(benchmark)):
            print(get_z3_command(i, file, result_dir))
    for benchmark in experiments.get("cvc5", []):
        result_dir = f"results/{benchmark}_cvc5"
        for i, file in enumerate(_get_files(benchmark)):
            print(get_cvc5_command(i, file, result_dir))

    for benchmark in experiments.get("nrago", []):
        result_dir = f"results/{benchmark}_nrago"
        for i, file in enumerate(_get_files(benchmark)):
            print(get_nrago_command(i, file, result_dir))

    for benchmark in experiments.get("ugotnl", []):
        result_dir = f"results/{benchmark}_ugotnl"
        for i, file in enumerate(_get_files(benchmark)):
            print(get_ugot_command(i, file, result_dir))

    for benchmark, configs in experiments.get("AttackModel", {}).items():
        for config in configs:
            for i, file in enumerate(_get_files(benchmark)):
                print(get_attacker_command(benchmark, i, file, config))

    if generate_custom_mbo:
        for num_summands in range(25, 76, 25):
            for N in range(1, 31, 1):
                for i in range(10):
                    print(get_mbo_generation_command(num_summands, N, i))

    if evaluate_custom_mbo:
        for summands in range(25, 99, 25):
            for N in range(1, 31, 1):
                for i in range(10):
                    result_dir = f"results/custom_mbo"
                    file = f"self_generated_mbo/mbo_{summands}_{N}_{i}.smt2"
                    file_preprocessed = f"preprocessed_mbo/self_generated_mbo/mbo_{summands}_{N}_{i}.smt2"
                    print(
                        get_z3_command(
                            f"mbo_{summands}_{N}_{i}_z3", file, result_dir, timeout=180
                        )
                    )
                    print(
                        get_attacker_command(
                            "mbo_preprocessed",
                            f"mbo_{summands}_{N}_{i}_attack",
                            file_preprocessed,
                            {
                                "l2o_version": "ours",
                                "forward_logic": "manual_mbo_preprocessed",
                            },
                            result_dir=result_dir,
                            timeout=180,
                        )
                    )


if __name__ == "__main__":
    main()
