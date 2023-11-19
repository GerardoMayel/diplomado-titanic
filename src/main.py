# Librerias
import pandas as pd
from titanic_transformers import fare_imputer, age_imputer, ticket_transformer
from sklearn.pipeline import Pipeline
from logging_config import setup_logging

setup_logging()

def main():
    # Cargar los datos del Titanic
    # Asegúrate de ajustar la ruta del archivo según sea necesario
    df = pd.read_csv('../data/phpMYEkMl-2.csv')

    # Crear la pipeline con los transformadores personalizados
    titanic_pipeline = Pipeline([
        ('fare_imputer', fare_imputer()),
        ('age_imputer', age_imputer()),
        ('ticket_transformer', ticket_transformer())
    ])

    # Aplicar las transformaciones
    df_transformed = titanic_pipeline.fit_transform(df)
    
    # Guardar el DataFrame transformado
    df_transformed.to_csv('../data/titanic_transformed.csv', index=False)

    return df_transformed

if __name__ == "__main__":
    transformed_df = main()
    print(transformed_df.head())  # Opcional: Mostrar las primeras filas del DataFrame transformado
