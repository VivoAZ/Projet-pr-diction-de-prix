import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


def predict_house_prices(file_path):
    """
    Pipeline complet pour charger, traiter et entraîner un modèle de régression sur des données de prix des maisons.

    Args:
        file_path (str): Chemin vers le fichier CSV contenant les données.

    Returns:
        tuple: Contient le modèle final entraîné et les scores MAE (entraînement et test).
    """
    # Charger les données
    df = pd.read_csv(file_path)
    print(f"Le jeu de données contient {df.shape[0]} lignes et {df.shape[1]} colonnes.")
    print(df.info())
    print(df.head())

    # Séparer les données en ensembles d'entraînement et de test
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=50)
    print("Informations sur l'ensemble d'entraînement :")
    df_train.info()
    print("Informations sur l'ensemble de test :")
    df_test.info()

    # Imputation des valeurs manquantes
    Col_nan_moy = ['CRIM', 'AGE', 'LSTAT']
    for colonne in Col_nan_moy:
        df_train[colonne] = df_train[colonne].fillna(df[colonne].mean())

    Col_nan_mode = ["ZN", "INDUS", "CHAS"]
    for colonne in Col_nan_mode:
        mode_value = df_train[colonne].mode()[0]
        df_train[colonne] = df_train[colonne].fillna(mode_value)

    # Visualisation des données
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['MEDV'])
    plt.title('Box plot de MEDV')
    plt.show()

    # Scatter plots
    features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(features):
        plt.subplot(4, 3, i + 1)
        sns.scatterplot(x=df[feature], y=df['MEDV'])
        plt.title(f'Scatter plot of {feature} vs MEDV')
    plt.tight_layout()
    plt.show()

    # Classe personnalisée pour la troncature des outliers
    class OutlierTruncator(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            X['CRIM'] = X['CRIM'].clip(upper=50)
            X['RAD'] = X['RAD'].clip(upper=10)
            X['TAX'] = X['TAX'].clip(upper=500)
            return X

    # Pipelines pour les transformations
    numeric_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'B', 'LSTAT']
    categorical_features = ['CHAS', 'RAD']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Modèles et sélection
    models = [
        DecisionTreeRegressor(random_state=50),
        RandomForestRegressor(random_state=50),
        GradientBoostingRegressor(random_state=50),
        XGBRegressor(n_estimators=100, max_depth=5),
        LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
    ]

    # Pipelines complets
    pipeline = Pipeline(steps=[
        ('outlier_truncator', OutlierTruncator()),
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])

    # Préparer les données pour le modèle
    X_train = df_train.drop(['MEDV', 'PTRATIO'], axis=1)
    y_train = df_train['MEDV']
    X_test = df_test.drop(['MEDV', 'PTRATIO'], axis=1)
    y_test = df_test['MEDV']

    # Entraîner et évaluer le modèle
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"MAE sur le jeu d'entraînement : {mae_train:.3f}")
    print(f"MAE sur le jeu de test : {mae_test:.3f}")

    # Enregistrement du modèle
    joblib.dump(pipeline, "gradient_boosting_pipeline.pkl")
    print("Modèle enregistré sous 'gradient_boosting_pipeline.pkl'.")

    return pipeline, mae_train, mae_test 

file_path = 'C:/Users/HP PROBOOK/Desktop/Projet prédiction de prix/HousingData.csv'
pipeline, mae_train, mae_test = predict_house_prices(file_path)

