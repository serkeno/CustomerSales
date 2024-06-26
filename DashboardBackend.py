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

import pandas as pd

from sklearn.inspection import PartialDependenceDisplay

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

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

    # Place csv data into a dataframe
    df = pd.read_csv("customer_shopping_data.csv")

    # Check to see if max_data_usage is higher than number of rows, using all rows if it is higher.
    if max_data_usage <= df.shape[0]:
        df = pd.read_csv("customer_shopping_data.csv").head(max_data_usage)

    # Replace 'Male' and 'Female' with 0 and 1
    # pd.set_option is used to opt into future behaviour, specifically to remove silent downcasting in .replace()
    pd.set_option('future.no_silent_downcasting', True)
    df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})

    del df["invoice_no"]
    del df["customer_id"]
    del df["invoice_date"]
    del df["shopping_mall"]

    onehot_encoder = OneHotEncoder()

    onehot_encoded_payment = onehot_encoder.fit_transform(df[['payment_method']]).toarray()
    onehot_encoded_payment_df = pd.DataFrame(onehot_encoded_payment, columns=onehot_encoder.get_feature_names_out(['payment_method']))

    onehot_encoded_category = onehot_encoder.fit_transform(df[['category']]).toarray()
    onehot_encoded_category_df = pd.DataFrame(onehot_encoded_category, columns=onehot_encoder.get_feature_names_out(['category']))

    df_encoded = pd.concat([df, onehot_encoded_payment_df, onehot_encoded_category_df], axis=1).drop(
        ['payment_method', 'category'], axis=1)

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

    p1 = Pipeline([('K-Neighbors Regression', KNeighborsRegressor())])

    param_grid = {
        'K-Neighbors Regression__n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider
        'K-Neighbors Regression__weights': ['uniform', 'distance'],  # Weight function used in prediction
        'K-Neighbors Regression__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
        'K-Neighbors Regression__leaf_size': [10, 20, 30],  # Leaf size passed to BallTree or KDTree
    }

    grid_search = GridSearchCV(estimator=p1, param_grid=param_grid, cv=5)

    fit_and_print(grid_search, X_train, y_train, X_test, y_test)

    return p1.fit(X_train, y_train)
def LR(max_data_usage=100):

    df = CustomerSales(max_data_usage=max_data_usage)

    X = df.drop('price', axis=1)
    X = preprocessor(X)

    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    p1 = Pipeline([('Linear Regression', LinearRegression())])

    param_grid = {'Linear Regression__fit_intercept': [True, False]}

    grid_search = GridSearchCV(estimator=p1, cv=5, param_grid=param_grid)

    fit_and_print(grid_search, X_train, y_train, X_test, y_test)

    return p1.fit(X_train, y_train)
def RF(max_data_usage=100):
    df = CustomerSales(max_data_usage=max_data_usage)

    X = df.drop('price', axis=1)
    X = preprocessor(X)

    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    p1 = Pipeline([('Random Forest', RandomForestRegressor())])

    param_grid = {
        'Random Forest__n_estimators': [50, 100, 200],
        'Random Forest__max_depth': [None, 10, 20],
        'Random Forest__min_samples_split': [2, 5, 10],
        'Random Forest__min_samples_leaf': [1, 2, 4],
        'Random Forest__max_features': ['auto', 'sqrt'],
        'Random Forest__bootstrap': [True, False],
        'Random Forest__random_state': [1],  # Set a specific random state for reproducibility
        'Random Forest__n_jobs': [-1]  # Use all available cores for parallel processing
    }

    grid_search = GridSearchCV(estimator=p1, param_grid=param_grid, cv=2)

    fit_and_print(grid_search, X_train, y_train, X_test, y_test)

    return p1.fit(X_train, y_train)
def fit_and_print(p, X_train, y_train, X_test, y_test):

    p.fit(X_train, y_train)
    train_preds = p.predict(X_train)
    test_preds = p.predict(X_test)

    print('Training error: ' + str(mean_absolute_error(train_preds, y_train)))
    print('Test error: ' + str(mean_absolute_error(test_preds, y_test)))


def single_prediction(fitted_model, record):
    """
    Takes a single record and a trained model or pipeline to perform prediction on, printing the result to console and
    returning it.
    :param fitted_model: Takes a trained model or pipeline.
    :param record: A single record supplied as a Numpy Array
    :return: Returns a ndarray of shape (n_samples), or (n_samples, n_targets) see sklearn.com .predict() docs for more
    info
    """
    single_record = record

    prediction = fitted_model.predict(single_record)

    print("Prediction:", prediction)

    return prediction
