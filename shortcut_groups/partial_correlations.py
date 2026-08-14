import numpy as np

# z1 and z2 are two residuals from the two different correlations we had
QUADRANT_KEYS = (
    "z1_negative_z2_negative",
    "z1_negative_z2_positive",
    "z1_positive_z2_negative",
    "z1_positive_z2_positive",
)

# residualisation of x1 and x2 wrt x3
def residualise(x1, x2, x3):
    # add an intercept column of ones to the left of x3
    mat_linalg = np.c_[np.ones(len(x3)), x3]
    # fit x1 and x2 as linear functions of x3
    # regression coefficient one
    beta1 = np.linalg.lstsq(mat_linalg, x1, rcond=None)[0]
    # regression coefficient two
    beta2 = np.linalg.lstsq(mat_linalg, x2, rcond=None)[0]
    return x1 - mat_linalg @ beta1, x2 - mat_linalg @ beta2


def residual_quadrants(z1, z2):
    return {
        "z1_negative_z2_negative": np.maximum(-z1, 0.0) * np.maximum(-z2, 0.0),
        "z1_negative_z2_positive": np.maximum(-z1, 0.0) * np.maximum(z2, 0.0),
        "z1_positive_z2_negative": np.maximum(z1, 0.0) * np.maximum(-z2, 0.0),
        "z1_positive_z2_positive": np.maximum(z1, 0.0) * np.maximum(z2, 0.0),
    }


def _compute_pcorrs_maps(bi, ba, pa, include_quadrants):
    """Correlate ``bi`` and ``pa`` after controlling for ``ba``."""
    maps = []
    quadrant_keys = QUADRANT_KEYS if include_quadrants else ()
    quadrants = {key: [] for key in quadrant_keys}
    pcorrs = []
    for r_bi, r_ba, r_pa in zip(bi, ba, pa):
        u, v = residualise(r_bi, r_pa, r_ba)
        # z_score the residuals
        u = (u - u.mean()) / (u.std(ddof=1) + 1e-12)
        v = (v - v.mean()) / (v.std(ddof=1) + 1e-12)
        # compute contributions of each region to obtain our contribution maps
        contribution = u * v
        maps.append(contribution)
        # if quadrants are required, the we can assign the contributions to each of the four quadrants
        if include_quadrants:
            values = residual_quadrants(u, v)
            for key in quadrant_keys:
                quadrants[key].append(values[key])
        # we are using contribution maps to compute correlation here
        score = float(contribution.sum() / (len(contribution) - 1))
        pcorrs.append(score)
    return (
        np.asarray(maps),
        np.asarray(pcorrs),
        {
            key: np.asarray(values)
            for key, values in quadrants.items()
        },
    )


def _build_shortcut_task_maps(bi, ba, pa, include_quadrants):
    # partial correlation is computed and decomposed as contribution maps for both shortcut and task partial correlation
    shortcut_maps, shortcut_scores, shortcut_quadrants = _compute_pcorrs_maps(
        bi,
        ba,
        pa,
        include_quadrants,
    )
    task_maps, task_scores, task_quadrants = _compute_pcorrs_maps(
        bi,
        pa,
        ba,
        include_quadrants,
    )
    # this is oscar kind of aggregate level partial correlation, just using median here for consistency
    aggregate_bi = np.median(bi, axis=0, keepdims=True)
    aggregate_ba = np.median(ba, axis=0, keepdims=True)
    aggregate_pa = np.median(pa, axis=0, keepdims=True)
    # compute the dataset level shortcut and task contribution maps now
    dataset_shortcut_maps = (
        _compute_pcorrs_maps(aggregate_bi, aggregate_ba, aggregate_pa, False)
    )[0]
    dataset_task_maps = (
        _compute_pcorrs_maps(aggregate_bi, aggregate_pa, aggregate_ba, False)
    )[0]

    result = {
        "shortcut_maps": shortcut_maps,
        "task_maps": task_maps,
        # only keeping positive evidence for most experiments except for the quadrant analysis
        "shortcut_evidence": np.maximum(shortcut_maps, 0.0),
        "task_evidence": np.maximum(task_maps, 0.0),
        "shortcut_scores": shortcut_scores,
        "task_scores": task_scores,
        "dataset_shortcut_evidence": np.maximum(dataset_shortcut_maps[0], 0.0),
        "dataset_task_evidence": np.maximum(dataset_task_maps[0], 0.0),
    }
    if include_quadrants:
        for key in QUADRANT_KEYS:
            result[f"shortcut_quadrant_{key}"] = shortcut_quadrants[key]
            result[f"task_quadrant_{key}"] = task_quadrants[key]
    return result

# this is our second level object i.e. contribution maps
def build_shortcut_task_maps(bi, ba, pa):
    return _build_shortcut_task_maps(bi, ba, pa, False)


def build_shortcut_task_maps_with_quadrants(bi, ba, pa):
    return _build_shortcut_task_maps(bi, ba, pa, True)
