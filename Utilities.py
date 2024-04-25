"""A list of utilities designed to clean up common Scikit-learn activities"""


def staggered_partial_dependence_displays(model, df, feature_indexes, max_buffer=3):
    """Creates Partial Dependence Displays at a given maximum per image to easily visualize multiple PDDs at once
    without having to tinker with display settings to make them fit.

    Parameters:
        model: The model used as estimator, will be loaded into the PartialDependenceDisplay.

        df: This is the DataFrame object that will be used to pull features from,
         will be loaded into the PartialDependenceDisplay

        feature_indexes: The indexes of the features you want to include in the staggered Partial Dependence Display.

        max_buffer: This is the maximum number of PDD charts displayed per image. Default=3

    Returns: None
    """
    from sklearn.inspection import PartialDependenceDisplay

    count = 0
    master_list = []
    current_list = []
    for feature in feature_indexes:

        current_list.append(feature)
        count += 1


        if count == max_buffer:
            master_list.append(current_list)
            current_list = []
            count = 0

    for feature_list in master_list:
        PartialDependenceDisplay.from_estimator(model, df, features=feature_list)


