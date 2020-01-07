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


