"""This is the backend dashboard for the Data Analysis Dashboard,
 which will act as the front end for this project. All functions for analyzing data in the attached CSV file will
 be written here, and only called within the Data Analysis Dashboard Jupyter Notebook file to create a layer of abstraction
 between the user and the functionality of the program.

 The following is requested as citation from the creators and contributors of Scikit-Learn:

 @article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}

 """

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def partial_dependence():
    # Use .drop(columns=['A', 'B', 'C']) to drop by column name or .drop([0, 1, 2]) to drop by index
    df = pd.read_csv("customer_shopping_data.csv").head(100)

    # Replace 'Male' and 'Female' with 0 and 1
    df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})

    del df["invoice_no"]
    del df["customer_id"]
    # del df["age"]
    del df["category"]
    # del df["quantity"]
    del df["payment_method"]
    del df["invoice_date"]
    del df["shopping_mall"]

    # Onehot encode gender, transform it, then concatenate with df and drop the original gender
    # Axis refers to whether you want to perform operations on rows or columns. Rows=0 Columns=1
    # ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
    # ohetransform = ohe.fit_transform(df[['gender']])
    # df = pd.concat([df, ohetransform], axis=1).drop(columns=['gender'])

    print(df.keys())

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    LR = LinearRegression().fit(X, y)

    PartialDependenceDisplay.from_estimator(LR, X, features=[0, 1, 2])
    # PartialDependenceDisplay.from_estimator(clf, X, features=[1])

def CustomerSales(max_data_usage=100):
    # Use .drop(columns=['A', 'B', 'C']) to drop by column name or .drop([0, 1, 2]) to drop by index
    df = pd.read_csv("customer_shopping_data.csv").head(max_data_usage)

    # Replace 'Male' and 'Female' with 0 and 1
    df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})

    del df["invoice_no"]
    del df["customer_id"]
    # del df["age"]
    del df["category"]
    # del df["quantity"]
    # del df["payment_method"]
    del df["invoice_date"]
    del df["shopping_mall"]

    onehot_encoder = OneHotEncoder()
    onehot_encoded = onehot_encoder.fit_transform(df[['payment_method']]).toarray()
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['payment_method']))
    df_encoded = pd.concat([df, onehot_encoded_df], axis=1).drop('payment_method', axis=1)

    return df_encoded
def preprocessor(X):
    std_scaler = StandardScaler().fit(X)
    min_max_scaler = MinMaxScaler().fit(X)

    return X
def KNR(max_data_usage=100):
    df = CustomerSales(max_data_usage=max_data_usage)

    X = df.drop('price', axis=1)
    X = preprocessor(X)

    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    p1 = Pipeline([('K-Neighbors Regression', KNeighborsRegressor(n_neighbors=7))])

    fit_and_print(p1, X_train, y_train, X_test, y_test)
def LR(max_data_usage=100):
    df = CustomerSales(max_data_usage=max_data_usage)

    X = df.drop('price', axis=1)
    X = preprocessor(X)

    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    p1 = Pipeline([('Linear Regression', LinearRegression())])

    fit_and_print(p1, X_train, y_train, X_test, y_test)
def RF():
    return 1
def fit_and_print(p, X_train, y_train, X_test, y_test):
    # Todo make functions for KNR, LR, and RF regressors to plug into this function
    #  and analyze the differences in performance
    #  Also, implement cross-validation for each to improve scoring
    p.fit(X_train, y_train)
    train_preds = p.predict(X_train)
    test_preds = p.predict(X_test)

    print('Training error: ' + str(mean_absolute_error(train_preds, y_train)))
    print('Test error: ' + str(mean_absolute_error(test_preds, y_test)))


