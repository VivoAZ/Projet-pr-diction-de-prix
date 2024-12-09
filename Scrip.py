import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

def load_data(file_path):
    """
    Charge le fichier CSV dans un DataFrame pandas.
    
    Parameters:
        file_path (str): Le chemin vers le fichier CSV.

    Returns:
        DataFrame: DataFrame pandas contenant les données du fichier CSV.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Préprocess les données en séparant le train et le test,
    et en imputant les valeurs manquantes pour certaines colonnes.

    Parameters:
        df (DataFrame): DataFrame pandas contenant les données brutes.

    Returns:
        DataFrame: DataFrame pré-traité avec les valeurs manquantes imputées.
    """
    # Séparation en train et test
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=50)

    # Imputation par la moyenne
    for col in ['CRIM', 'AGE', 'LSTAT']:
        df_train[col] = df_train[col].fillna(df[col].mean())

    # Imputation par le mode
    for col in ["ZN", "INDUS", "CHAS"]:
        mode_value = df_train[col].mode()[0]
        df_train[col] = df_train[col].fillna(mode_value)

    # Troncatuer les valeurs extrêmes
    df_train['CRIM'] = df_train['CRIM'].clip(upper=50)
    df_train['RAD'] = df_train['RAD'].clip(upper=10)
    df_train['TAX'] = df_train['TAX'].clip(upper=500)

    return df_train, df_test

def create_pipeline():
    """
    Crée un pipeline de préprocessing complet avec une classe personnalisée
    pour la troncature des outliers.

    Returns:
        Pipeline: Pipeline sklearn avec tous les étapes de préprocessing.
    """
    class OutlierTruncator(BaseEstimator, TransformerMixin):
        """
        Transformer pour troncature des outliers dans les données.
        """
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X['CRIM'] = X['CRIM'].clip(upper=50)
            X['RAD'] = X['RAD'].clip(upper=10)
            X['TAX'] = X['TAX'].clip(upper=500)
            return X

    # Définir les colonnes numériques et catégorielles
    numeric_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'B', 'LSTAT']
    categorical_features = ['CHAS', 'RAD']

    # Pipeline pour les caractéristiques numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline pour les caractéristiques catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combiner les transformateurs
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Créer le pipeline complet avec l'étape de troncature des outliers
    pipeline = Pipeline(steps=[
        ('outlier_truncator', OutlierTruncator()),
        ('preprocessor', preprocessor)
    ])

    return pipeline

def train_models(X_train, y_train):
    """
    Entraîne plusieurs modèles de régression et évalue leur performance par cross-validation.

    Parameters:
        X_train (DataFrame): Données d'entraînement.
        y_train (Series): Cibles d'entraînement.

    Returns:
        tuple: Meilleur modèle selon la MAE et son score MAE.
    """
    models = [
        DecisionTreeRegressor(random_state=50),
        RandomForestRegressor(random_state=50),
        GradientBoostingRegressor(random_state=50),
        XGBRegressor(n_estimators=100, max_depth=5),
        LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
    ]

    mae_scores = []

    for model in models:
        y_train_pred = cross_val_predict(model, X_train, y_train, cv=3, n_jobs=-1)
        mae_scores.append(mean_absolute_error(y_train, y_train_pred))

    best_model_idx = np.argmin(mae_scores)
    best_model_mae = models[best_model_idx].__class__.__name__
    best_model_mae_score = mae_scores[best_model_idx]

    return best_model_mae, best_model_mae_score

def main():
    # Charger les données
    file_path = 'C:/Users/HP PROBOOK/Desktop/Projet prédiction de prix/HousingData.csv'
    df = load_data(file_path)

    # Préprocess les données
    df_train, df_test = preprocess_data(df)

    # Créer le pipeline de préprocessing
    pipeline = create_pipeline()

    # Séparer les features et la cible
    X_train = df_train.drop(['MEDV', 'PTRATIO'], axis=1)
    y_train = df_train['MEDV']
    X_test = df_test.drop(['MEDV', 'PTRATIO'], axis=1)

    # Entraîner les modèles
    best_model_mae, best_model_mae_score = train_models(X_train, y_train)

    print(f"Meilleur modèle selon la MAE : {best_model_mae} (MAE={best_model_mae_score:.3f})")

    # Enregistrer le modèle
    joblib.dump(models[best_model_idx], "gradient_boosting_model.pkl")

    # Charger le modèle pour vérification
    loaded_model = joblib.load("gradient_boosting_model.pkl")

if __name__ == "__main__":
    main()
