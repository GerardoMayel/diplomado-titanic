# librerias
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logging_config import setup_logging
import logging

setup_logging()

# Obtener una instancia del logger
logger = logging.getLogger(__name__)

# Definiciones de los transformadores personalizados


#funcion generan para eliminar el signo de interrogacion en los datos
def replace_question_with_nan(X, column):
    try:
        X[column] = X[column].replace('?', pd.NA)
        return X
    except Exception as e:
        logger.error(f"titanic_transformers.py : Error al reemplazar '?' por NaN en {column}: {e}")
        raise

#Tarifa
        
class fare_imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = replace_question_with_nan(X, 'fare')
        self.mean_fare = pd.to_numeric(X['fare'], errors='coerce').mean()
        return self

    def transform(self, X, y=None):
        try:
            logger.info("titanic_transformers.py : Transformando datos con fare_imputer...")
            X = replace_question_with_nan(X, 'fare')
            X['fare'] = pd.to_numeric(X['fare'], errors='coerce')
            X['fare'].fillna(self.mean_fare, inplace=True)
            return X
        except Exception as e:
            logger.error(f"titanic_transformers.py : Error en fare_imputer: {e}")
            raise

#Edad
            
class age_imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = replace_question_with_nan(X, 'age')
        self.median_age = pd.to_numeric(X['age'], errors='coerce').median()
        return self

    def transform(self, X, y=None):
        try:
            logger.info("titanic_transformers.py : Transformando datos con age_imputer...")
            X = replace_question_with_nan(X, 'age')
            X['age'] = pd.to_numeric(X['age'], errors='coerce')
            X['age'].fillna(self.median_age, inplace=True)
            return X
        except Exception as e:
            logger.error(f"titanic_transformers.py : Error en age_imputer: {e}")
            raise

#Ticket
            
class ticket_transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            logger.info("titanic_transformers.py : Transformando datos con ticket_transformer...")
            X = replace_question_with_nan(X, 'ticket')
            X['ticket'] = X['ticket'].str.extract('(\d+)', expand=False)
            X['ticket'] = pd.to_numeric(X['ticket'], errors='coerce')
            X['ticket'].fillna(0, inplace=True)
            return X
        except Exception as e:
            logger.error(f"titanic_transformers.py : Error en ticket_transformer: {e}")
            raise

