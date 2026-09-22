# eval_test.py

from MLstatkit import Bootstrapping


def get_reduced_setup(residual_setup):
    if residual_setup not in ["SQT", "ST", "QT"]:
        raise ValueError(f"Unknown residual_setup: {residual_setup}")
    return residual_setup[:-1]


def get_full_reduced_scores(
    residual_setup, pred_df, y_true, n_bootstraps, seed, show_progress=False
):
    """
    Get full and reduced scores for a given residual setup, along with delta scores and confidence intervals.
    """
    if residual_setup not in ["SQT", "ST", "QT", "T"]:
        raise ValueError(f"Unknown residual_setup: {residual_setup}")
    ## Get full score
    full_name = "pred_stage3_total"
    full_score, full_ci_lo, full_ci_hi = Bootstrapping(
        y_true=y_true,
        y_prob=pred_df[full_name],
        metric_str="r2",
        n_bootstraps=n_bootstraps,
        confidence_level=0.95,
        show_progress=show_progress,
        random_state=seed,
    )
    if residual_setup == "T":
        return {
            "single_score": {
                residual_setup: f"{full_score:.3f} ({full_ci_lo:.3f}, {full_ci_hi:.3f})"
            }
        }
    ## Get reduced score
    # Stage 2 for full-reduced SQT/SQ and QT/Q models-- stage 1 for ST/S models
    reduced_name = (
        "pred_stage2_total" if residual_setup in ["SQT", "QT"] else "pred_stage1"
    )
    reduced_setup = get_reduced_setup(residual_setup)
    reduced_score, reduced_ci_lo, reduced_ci_hi = Bootstrapping(
        y_true=y_true,
        y_prob=pred_df[reduced_name],
        metric_str="r2",
        n_bootstraps=n_bootstraps,
        confidence_level=0.95,
        show_progress=show_progress,
        random_state=seed,
    )
    # Get delta score and CI
    _, _, diff, ci_lo_diff, ci_hi_diff = Bootstrapping(
        y_true=y_true,
        y_prob=pred_df[full_name],
        y_prob_b=pred_df[reduced_name],
        metric_str="r2",
        n_bootstraps=n_bootstraps,
        confidence_level=0.95,
        show_progress=show_progress,
        random_state=seed,
    )
    return {
        "single_score": {
            residual_setup: f"{full_score:.3f} ({full_ci_lo:.3f}, {full_ci_hi:.3f})",
            reduced_setup: f"{reduced_score:.3f} ({reduced_ci_lo:.3f}, {reduced_ci_hi:.3f})",
        },
        "delta_score": {
            f"{residual_setup}-{reduced_setup}": f"{diff:.3f} ({ci_lo_diff:.3f}, {ci_hi_diff:.3f})"
        },
    }


def aggregate_result_lists(
    single_scores,
    paired_scores,
    train_score_dict,
    test_score_dict,
    residual_setup,
    impact_type,
):
    """
    Aggregate result lists for single and paired scores.

    Params
    -----
    single_scores : list
        List to store single score results.
    paired_scores : list
        List to store paired score results.
    train_score_dict : dict
        Dictionary containing training scores.
    test_score_dict : dict
        Dictionary containing test scores.
    residual_setup : str
        One of ['SQT', 'ST', 'QT', 'T'] indicating the residual setup.
    impact_type : str
        Type of impact being evaluated.
    """
    ## SINGLE SCORES
    for set_up in train_score_dict["single_score"].keys():
        assert (
            set_up in test_score_dict["single_score"]
        ), f"Set-up type {set_up} not found in test scores"
        # Append results
        single_scores.append(
            {
                "residual_setup": set_up,
                "impact_type": impact_type,
                "r2_train": train_score_dict["single_score"][set_up],
                "r2_test": test_score_dict["single_score"][set_up],
            }
        )
    ## PAIRED SCORES
    if residual_setup in ["SQT", "ST", "QT"]:
        for set_up in train_score_dict["delta_score"].keys():
            assert (
                set_up in test_score_dict["delta_score"]
            ), f"Set-up type {set_up} not found in test scores"
            paired_scores.append(
                {
                    "full/reduced": set_up,
                    "impact_type": impact_type,
                    "r2_train": train_score_dict["delta_score"][set_up],
                    "r2_test": test_score_dict["delta_score"][set_up],
                }
            )
    return single_scores, paired_scores
