from Attack_models import (
    sequential,
    gpt_kissing,
    gpt_mbo,
    manual_kissing,
    manual_mbo_orig,
    manual_mbo_prepr,
)
import parse_smt2

attackModelClass = {
    "manual_kissing": manual_kissing.ManualKissing,
    "manual_mbo_preprocessed": manual_mbo_prepr.ManualMboPreprocessed,
    "manual_mbo_orig": manual_mbo_orig.ManualMboOrig,
    "gpt_kissing": gpt_kissing.GPTKissing,
    "gpt_mbo_preprocessed": gpt_mbo.GPTMbo,
    "sequential": sequential.Sequential,
}


def create_my_model(path, epsilon, device, l2o_version, forward_logic):
    script = parse_smt2.get_formula(path)
    AttackModel = attackModelClass[forward_logic]

    model = AttackModel(
        path,
        script,
        epsilon=epsilon,
        device=device,
        l2o_version=l2o_version,
    )
    input_dim = model.get_dimension_input()
    return model, input_dim, script
