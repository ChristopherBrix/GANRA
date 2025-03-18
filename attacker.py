import create_my_model
import utils

import argparse
import os
import random
from typing import TypeVar
import time
import torch
import numpy as np
import z3
from tqdm import tqdm
import pytorch_optimizer


T = TypeVar("T")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class OwnConstraints:
    def __init__(self, forward_logic, script):
        self.forward_logic = forward_logic
        variables = utils.get_variables(script)
        self.input_array = variables

        self.reset()

    def reset(self):
        OwnConstraints.prev_logits = 0
        OwnConstraints.abort = False

    def __call__(self, input: T, all_constraints_satisfied) -> T:
        results = torch.zeros((1, input.shape[0]), dtype=torch.bool)
        if (
            OwnConstraints.abort
        ):  # abortion of attack since there logits were not changing anymore
            print("Aborting attack.", flush=True)
            results[0] = torch.ones_like(results)
            return results, -1
        if all_constraints_satisfied.any().item():
            results[0] = torch.ones_like(results)
            true_indices = torch.nonzero(all_constraints_satisfied).reshape(-1)
            return (results, true_indices[0].item())

        return results, -1


def get_loss_fn(model):
    def loss_fn(inputs):
        logits, add_information = model(inputs)
        # no change during the attack triggers a reset
        if (logits == OwnConstraints.prev_logits).all():
            OwnConstraints.abort = True
        OwnConstraints.prev_logits = (
            logits  # store current logits to detect if no changes take place
        )
        return logits.sum(), add_information

    return loss_fn


def write_smt_solution_to_file(
    filename, orig_problem_path, solution, search_epsilon, model
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        with open(orig_problem_path, "r") as orig_file:
            orig_lines = orig_file.readlines()
        while len(orig_lines) > 0:
            if orig_lines[0].startswith("(assert"):
                break
            f.write(orig_lines.pop(0))
        for i in range(solution.shape[0]):
            f.write(
                f"(assert (>= {solution[i].item() + search_epsilon:.10f} {model.inputs[i]}))\n"
            )
            f.write(
                f"(assert (<= {solution[i].item() - search_epsilon:.10f} {model.inputs[i]}))\n"
            )
        f.write("\n")
        while len(orig_lines) > 0:
            f.write(orig_lines.pop(0))


def is_candidate_solution_valid(orig_problem_path, data, search_epsilon, model):
    sat_filename = f"/tmp/test_{time.time()}.smt2"
    write_smt_solution_to_file(
        sat_filename, orig_problem_path, data, search_epsilon, model
    )
    solver = z3.Solver()
    constraints = z3.parse_smt2_file(sat_filename)
    solver.add(constraints)
    solver.set("timeout", 10 * 1000)
    r = solver.check()
    if r == z3.sat:
        return True
    elif r == z3.unsat:
        return False
    else:
        print(f"Z3 could not verify the potential solution. Result is {r}.")
        return False


def attack(image, constraint, steps, step_size, loss_fn, attack_type):
    index = -1
    image = torch.autograd.Variable(image, requires_grad=True)
    if attack_type == "adam":
        optimizer = torch.optim.Adam([image], lr=step_size)
    elif attack_type == "sgd":
        optimizer = torch.optim.SGD([image], lr=step_size)
    elif attack_type == "signsgd":
        optimizer = pytorch_optimizer.SignSGD([image], lr=step_size, momentum=0.0)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    for steps in tqdm(range(steps)):
        optimizer.zero_grad()
        loss, all_constraints_satisfied = loss_fn(image)
        is_adv, index = constraint(image, all_constraints_satisfied)
        if is_adv.any():
            break

        loss.backward()
        optimizer.step()
        image.data = torch.clamp(image.data, -1.0, 1.0)
    print("There were ", steps, " points needed. ")
    return image, index


model = None


def main() -> None:
    global model
    parser = argparse.ArgumentParser(
        description="NRA solver using adversarial attacks."
    )
    parser.add_argument("path", type=str, help="Path to the benchmark file.")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=2,
        default=(-1.0, 1.0),
        help="Bounds for the adversarial example.",
    )
    parser.add_argument(
        "--init_rand_bounds",
        type=float,
        nargs=2,
        default=(-1.0, 1.0),
        help="Bounds for the initial random value.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for the adversarial attack.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="adam",
        help="Attack method to use for the adversarial attack.",
        choices=["sgd", "signsgd", "adam"],
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps for the adversarial attack.",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=1e-3,
        help="Absolute step size for the adversarial attack.",
    )
    parser.add_argument(
        "--search_epsilon",
        type=float,
        default=1e-4,
        help="Defines half of the search space in which z3 is looking for an adversarial example.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0001,
        help="Epsilon area around (in)equalities during gradient computation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU acceleration.",
    )
    parser.add_argument(
        "--sat_instance_output_file",
        type=str,
        default=None,
        help="Path to save the found adversarial example.",
    )
    parser.add_argument(
        "--l2o_version",
        type=str,
        default="ours",
        choices=["ours", "ugotnl", "nrago"],
        help="What L2O definition to use.",
    )
    parser.add_argument(
        "--forward_logic",
        type=str,
        choices=[
            "manual_kissing",
            "manual_mbo_preprocessed",
            "manual_mbo_orig",
            "gpt_kissing",
            "gpt_mbo_preprocessed",
            "sequential",
        ],
        help="What code base to use to compute the forward pass for gradient descent",
    )
    args = parser.parse_args()

    path = args.path
    bounds = tuple(args.bounds)
    init_rand_bounds = tuple(args.init_rand_bounds)
    batch_size = args.batch_size
    attack_type = args.attack
    steps = args.steps
    step_size = args.step_size
    epsilon = args.epsilon
    seed = args.seed
    search_epsilon = args.search_epsilon
    gpu = args.gpu
    sat_instance_output_file = args.sat_instance_output_file
    l2o_version = args.l2o_version
    forward_logic = args.forward_logic
    print(args)

    if gpu:
        device = "cuda"
    else:
        device = "cpu"

    if seed is not None:
        _set_random_seed(seed)

    # create model with the given parameters,
    model, input_dim, script = create_my_model.create_my_model(
        path,
        epsilon,
        device=device,
        l2o_version=l2o_version,
        forward_logic=forward_logic,
    )
    print(f"Input_variables are: {model.variables}")
    model.eval()

    image = (
        torch.rand((batch_size, input_dim), device=device)
        * (init_rand_bounds[1] - init_rand_bounds[0])
        + init_rand_bounds[0]
    )

    constraint = OwnConstraints(forward_logic=forward_logic, script=script)
    while True:
        constraint.reset()

        image, index = attack(
            image,
            constraint,
            steps=steps,
            step_size=step_size,
            loss_fn=get_loss_fn(model),
            attack_type=attack_type,
        )

        if index != -1:  # satisfying result has been found
            is_sat = False
            need_to_check_with_z3 = (
                ("mbo_orig" in forward_logic and epsilon > 0)
                or "gpt" in forward_logic
                or "kissing" in forward_logic
                or l2o_version != "ours"
            )
            if need_to_check_with_z3:
                is_sat = is_candidate_solution_valid(
                    path, image[index], search_epsilon, model
                )
            else:
                is_sat = True

            if is_sat:
                if sat_instance_output_file is not None:
                    write_smt_solution_to_file(
                        sat_instance_output_file,
                        path,
                        image[index],
                        search_epsilon,
                        model,
                    )
                print("sat")
                break
            else:
                print("Potential candidate is not a solution.")
                image = (
                    torch.rand((batch_size, input_dim), device=device)
                    * (init_rand_bounds[1] - init_rand_bounds[0])
                    + init_rand_bounds[0]
                )
                continue
        optimized_inputs = image
        corresponding_outputs, _ = model(optimized_inputs)
        sorted_indices = torch.argsort(corresponding_outputs, descending=False)

        # Randomize the best half of the inputs and try again
        best_third = optimized_inputs[sorted_indices[: len(sorted_indices) // 3]]
        best_third_plus_noise = best_third + torch.randn_like(best_third) * 0.01
        random_third = (
            torch.rand((batch_size // 3, input_dim), device=device)
            * (init_rand_bounds[1] - init_rand_bounds[0])
            + init_rand_bounds[0]
        )
        image = torch.cat((best_third, best_third_plus_noise, random_third), dim=0)
        image = torch.clip(image, bounds[0], bounds[1])


if __name__ == "__main__":
    main()
