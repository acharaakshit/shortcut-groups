import numpy as np
from joblib import Memory
from sklearn.decomposition import NMF

# used for k-means
def row_l2_normalize(x):
    # take the norm of each row
    norms = np.linalg.norm(x, axis=1)
    # create array with same shape as x but filled with zeros
    out = np.zeros_like(x)
    # find zero rows first because they have no cluster information
    zero_rows = norms <= 1e-12
    # get the non-zero rows
    informative = ~zero_rows
    # take these rows, fill them with the original x values and apply l2 normalisation
    out[informative] = x[informative] / norms[informative, None]
    return out, zero_rows

# use for NMF functions
def row_simplex_normalize(x):
    # take the sum for each row in x
    row_sums = x.sum(axis=1)
    # create array with same shape as x but filled with zeros
    out = np.zeros_like(x)
    # find zero rows first because we need to assign zero memberships to them later
    zero_rows = row_sums <= 1e-12
    # get the non-zero rows
    informative = ~zero_rows
    # normalise the non-zero rows
    out[informative] = x[informative] / row_sums[informative, None]
    return out, zero_rows

# k-means is hard clustering so we need one-hot membership
def one_hot(labels, num_clusters):
    # take the cluster labels and make one dimensional
    labels = labels.reshape(-1)
    # create a test size times no. of cluster matrix filled with zeroes
    memberships = np.zeros((len(labels), num_clusters), dtype=float)
    # assign full memberships to one column in each row
    memberships[np.arange(len(labels)), labels] = 1.0
    return memberships

# standard k-means fit
def fit_kmeans(x, num_clusters):
    try:
        from cuml.cluster import KMeans
    except ImportError as error:
        raise ImportError(
            "K-means experiments require a compatible RAPIDS cuML install."
        ) from error
    model = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=50,
        random_state=0,
        output_type="numpy",
    ).fit(x)
    return model.labels_, model

# this is just to build the shortcut/task/joint representations
def build_shortcut_group_features(maps, feature):
    shortcut = maps["shortcut_evidence"]
    task = maps["task_evidence"]

    if feature == "joint":
        return np.concatenate([shortcut, task], axis=1)
    shortcut_signed = maps["shortcut_maps"]
    task_signed = maps["task_maps"]
    # each row basically has four values for every region now so the shape should be test size * (4 * grid regions)
    return np.concatenate(
        [
            np.maximum(shortcut_signed, 0.0), # positive shortcut
            np.maximum(-shortcut_signed, 0.0), # negative shortcut
            np.maximum(task_signed, 0.0), # positive task
            np.maximum(-task_signed, 0.0), # negative task
        ],
        axis=1,
    )

# this is to fit NMF, cache is enable to avoid refitting and save compute
@Memory(".cache/nmf_fits", verbose=0).cache
def fit_nmf_mixture(x_fit, num_components):
    # full matrix, zero rows
    x_fit, zero_rows = row_simplex_normalize(x_fit)
    # get non-zero rows for fitting
    informative = ~zero_rows
    # initialise the NMF model
    model = NMF(
        n_components=num_components, # number of shortcut groups
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        random_state=0,
        max_iter=2000,
    )
    # fit the NMF on our fitting rows
    fit_weights = model.fit_transform(x_fit[informative])

    # make sure the memberships sum to one after correctly scaling NMF components
    informative_memberships = scale_corrected_nmf_memberships(
        fit_weights,
        model.components_,
    )
    # zero matrix of the fit times no of shortcut groups shape
    memberships = np.zeros((len(x_fit), num_components), dtype=float)
    # fill the actual memberships, leave others zero
    memberships[informative] = informative_memberships
    # relative squared reconstruction error
    return memberships, model.reconstruction_err_, model

def scale_corrected_nmf_memberships(weights, components):
    """Return soft memberships."""
    # compute row level sum for each component
    nmf_component_mass = components.sum(axis=1)
    scaled_weights = weights * nmf_component_mass[None, :]
    return row_simplex_normalize(scaled_weights)[0]

# this is applied on the held-out sets for applying NMF
def transform_nmf_mixture(model, x):
    """Transform new samples using NMF fitting."""
    x, zero_rows = row_simplex_normalize(x)
    informative = ~zero_rows
    memberships = np.zeros((len(x), model.n_components), dtype=float)
    if np.any(informative):
        weights = model.transform(x[informative])
        memberships[informative] = scale_corrected_nmf_memberships(
            weights,
            model.components_,
        )
    return memberships
