import numpy as np


def mmpmr(s, threshold=0.6):
    """
    Computes the Mated Morph Presentation Match Rate or MMPMR given a matrix of euclidean distances of shape M x 2
    :param s: A numpy array with shape (M, 2), where M is the number of morphs.
        If the morph is not detected as a face by the model, a nan value can be used. This is counted as a failed morph.
    :param threshold: The threshold that defines how much distance between faces is still considered a match.
    :return: The MMPMR
    """

    # Since the input is euclidean distances rather than similarity, we use the maximal distance.
    maximum_distances = np.max(s, axis=1)

    # Replace the NaN values with failing values to avoid a RuntimeWarning when using < on NaN values
    maximum_distances[np.isnan(maximum_distances)] = threshold + 1.0

    below_threshold = maximum_distances < threshold

    # Make an array containing 0 for failed morphs and 1 for successful morphs
    successful_morphs = np.zeros_like(maximum_distances)
    successful_morphs[below_threshold] = 1
    return np.sum(successful_morphs)/successful_morphs.shape[0]


def relative_morph_distance(dist_x1_morph, dist_x2_morph, dist_x1_x2):
    """
    As far as I am aware, this is a metric thought of.

    WARNING: Consider that this metric is used mostly on embeddings from models that use a hypersphere as embedding space!
    This means that even the best model will not be able to acquire a score of 1 since the space is curved and
    therefore the morph can never lie exactly on a direct line between x1 and x2, unless x1 == x2.

    If a morph is perfectly in-between x1 and x2, then the max(dist_x1_morph, dist_x2_morph) == dist_x1_x2.
        Based on this idea, this metric measures the how big max(dist_x1_morph, dist_x2_morph) is relative to dist_x1_x2
        In the best case this value is 1.
        This metric does not like NaN values
    :param dist_x1_morph: Euclidean distances from the x1s to the morphs
    :param dist_x2_morph: Euclidean distances from the x2s to the morphs
    :param dist_x1_x2: Euclidean distances from the x1s to the x2s
    :return: (Mean RMD, RMD values for indices)
    """

    rmd_values = np.maximum(dist_x1_morph, dist_x2_morph)/(0.5*dist_x1_x2)
    return rmd_values.mean(), rmd_values
