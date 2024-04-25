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
import seaborn.objects as so

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def spending_by_category_by_gender():

    df = pd.read_csv("customer_shopping_data.csv")
    male_spending_credit = df.groupby(["gender", "payment_method"])
    # male_spending_debit
    # male_spending_cash

    # female_spending_credit
    # female_spending_debit
    # female_spending_cash
    sns.catplot(data=male_spending_credit, x="gender", y="payment_method", kind="count")


def spending_by_payment_method():
    df = pd.read_csv("customer_shopping_data.csv")

    sns.catplot(data=df, x="payment_method", y="price", kind="bar")


def spending_by_gender_swarm():
    df = pd.read_csv("customer_shopping_data.csv").head(1000)

    # sns.catplot(data=df, x="gender", y="price", kind="swarm")
    # sns.stripplot(data=df, x="gender", y="price")
    sns.displot(data=df, x="gender", y="price")


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


def partial_dependence_payment_method():
    # Use .drop(columns=['A', 'B', 'C']) to drop by column name or .drop([0, 1, 2]) to drop by index
    df = pd.read_csv("customer_shopping_data.csv").head(100)

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
    print(df_encoded.keys())

    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    LR = LinearRegression().fit(X, y)

    PartialDependenceDisplay.from_estimator(LR, X, features=[0, 1, 2, 3, 4, 5])


def partial_dependence_using_df2():

    import DataFrame2
    from Utilities import staggered_partial_dependence_displays

    # Use .drop(columns=['A', 'B', 'C']) to drop by column name or .drop([0, 1, 2]) to drop by index
    df2 = DataFrame2.DataFrame2()
    df2.set_inner_frame(updated_frame=pd.read_csv("customer_shopping_data.csv").head(100))
    df = df2.get_inner_frame()
    # Replace 'Male' and 'Female' with 0 and 1
    df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})

    del df["invoice_no"]
    del df["customer_id"]
    # del df["age"]
    # del df["category"]
    # del df["quantity"]
    # del df["payment_method"]
    del df["invoice_date"]
    del df["shopping_mall"]

    df_encoded = df2.multi_onehot_encoding(["category", "payment_method"], drop=True)

    print(df_encoded.keys())

    X = df_encoded.drop('price', axis=1)
    y = df_encoded['price']

    # Todo df_encoded = df2.create_variables(y) returns X as all but y parameter in df2 features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    LR = LinearRegression().fit(X, y)

    # Create a Partial Dependence Display of all features in the X dataframe.
    features = list(range(len(X.columns)))
    # PartialDependenceDisplay.from_estimator(LR, X, features=features)
    staggered_partial_dependence_displays(LR, X, feature_indexes=features)
