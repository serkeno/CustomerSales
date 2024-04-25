
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""This is an upgraded version of the Pandas Dataframe, including new methods for quality of life changes using
the Composition method of class extension. It creates a new class, DataFrame2, which will be used in place of pd.DataFrame
and provide additional functionality."""


class DataFrame2:

    def __init__(self):
        self.inner_frame = pd.DataFrame()

    def get_inner_frame(self):
        return self.inner_frame

    def set_inner_frame(self, updated_frame):
        self.inner_frame = updated_frame

    def multi_onehot_encoding(self, features, drop=True):
        """
            Convert multiple data frame columns into Onehot Encoded columns

            Parameters:
                features: The name of each feature you want to onehot encode

                drop: Determines if the original columns will be automatically removed from the df. True will
                remove the original, False will keep them. Default set to True.

            Returns: df, a copy of the modified dataframe after being transformed.


        """
        # Create a copy of self to modify
        df = self.inner_frame

        # Take the names of features you want to one_hot encode and check them for validity
        for feature in features:

            if feature not in features:
                raise ValueError("Feature " + feature + " not found in DataFrame Keys")

            onehot_encoder = OneHotEncoder()
            onehot_encoded = onehot_encoder.fit_transform(df[[feature]]).toarray()
            onehot_encoded_df = pd.DataFrame(onehot_encoded,columns=onehot_encoder.get_feature_names_out([feature]))

            if drop is True:
                df = pd.concat([df, onehot_encoded_df], axis=1).drop(feature, axis=1)
            else:
                df = pd.concat([df, onehot_encoded_df], axis=1)

        return df


