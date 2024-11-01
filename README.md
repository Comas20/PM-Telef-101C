# PM-Telef-101C
Repository including the mobility data provided by telefonica, the siniestrality data, and meteorologic data that we are going to use , the code that we will apply to clean the data and the algorithm to train our predictive model.

To install the materials, create a virtual environment and install the dependencies.
On Mac/Linux this will be:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## DEPURATION dir
Inside the __DEPURATION__ directory, we have the following items:

- DATA Directory: This directory contains our original datasets, which include raw data regarding mobility and accidents from 2023. The data within this folder has not been modified and is kept intact for reference and future use.

- Cleaned Data xlsx File: This Excel file contains the cleaned and processed data, derived from the original datasets in the DATA directory. The cleaning process involves removing inaccuracies, handling missing values, and restructuring the data to make it suitable for analysis.
  
- Jupyter Notebook: This notebook provides a detailed overview of the data cleaning and merging process we employed to obtain the cleaned dataset. It includes the code used, along with explanations of each step taken during the data preparation phase. The notebook serves as a valuable resource for anyone looking to replicate our methodology or apply similar processes to different datasets.

Please note that the Jupyter Notebook is included as documentation of our work and also as a resource for merging different datasets in the future. We have already applied this process to our raw data related to mobility and accidents from 2023, users can leverage the notebook to adapt the cleaning and merging techniques to their own datasets as needed.

### DATA dir
In the __DATA directory__, we have two main CSV files and one auxiliary file that is essential for obtaining the cleaned data.

- `ACCIDENTES_METEO-TABLA.csv`: This file contains detailed data regarding accidents that occurred in various provinces throughout the year. For each day, it contains information about the number of accidents in each province, with the corresponding meteorological conditions on that day. This dataset is crucial for understanding how weather influences accident rates.
  
- `movilidad_provincias_2023.csv`: This file provides comprehensive information on the number of displacements and travelers moving from one province to another each day of the year. It includes metrics such as the total number of trips made and the origin and destination provinces. This dataset is necessary for analyzing mobility patterns and understanding how travel behavior may correlate with accident occurrences.
  
- Auxiliary File: An additional auxiliary file is included in this directory to facilitate the cleaning and merging processes. This file is not directly involved in the final analysis, It contains a cleaned version of the `ACCIDENTES_METEO-TABLA.csv` file that will be use to merge it with the `movilidad_provincias_2023.csv` file.
  
Together, these datasets form the backbone of our analysis, they providing the necessary information to examine the relationships between mobility, accidents, and weather conditions.

### MergingInfo.ipynb
In this notebook, we implement a series of transformations on the two CSV files mentioned earlier to clean our data, preparing it for the training of our predictive model. The code is organized into two main sections.

First, we focus on cleaning the `ACCIDENTES_METEO-TABLA.csv` file. This initial step involves several preprocessing techniques to ensure that the data is accurate and suitable for analysis. The following code illustrates this process:

```python
# Importamos pandas
import pandas as pd

# Cargamos el archivo CSV (modifica 'nombre_del_archivo.csv' por el nombre de tu archivo)
archivo_csv = '../DEPURATION/DATA/ACCIDENTES_METEO - TABLA.csv'  # Cambia a tu nombre de archivo
df = pd.read_csv(archivo_csv)

# Revisamos si hay valores nulos en las columnas claves y los mostramos, si existen
print("Valores nulos en columnas de agrupación:")
print(df[['MES', 'DIA_MES', 'COD_PROVINCIA']].isnull().sum())

# Eliminamos filas con valores nulos en las columnas de agrupación
df = df.dropna(subset=['MES', 'DIA_MES', 'COD_PROVINCIA'])

# Agrupamos por MES, DIA_SEMANA, COD_PROVINCIA, calculando las agregaciones solicitadas
resultado = df.groupby(['MES', 'DIA_MES', 'COD_PROVINCIA'], as_index=False).agg(
    TOTAL_VICTIMAS=('TOTAL VICTIMAS', 'sum'),                # Sumar el total de víctimas
    CONDICION_METEO=('CONDICION_METEO', lambda x: x.mode()[0] if not x.mode().empty else None)  # Condición climática más frecuente
)

# Guardamos el resultado en un archivo Excel
resultado.to_excel('../DEPURATION/DATA/resultado_agrupado.xlsx', index=False)

# Mostramos el resultado
resultado.head()
```

In the second part of the notebook, we merge the cleaned `ACCIDENTES_METEO-TABLA.csv` data obtained from the previous code with the `movilidad_provincias_2023.csv` file. This step is essential to create a understandable dataset that will be used to train our predictive model. The following code snippet illustrates how this merging process is performed:

```python
# Cargamos el primer archivo (datos previos con información de víctimas y meteorología) desde Excel
archivo_victimas = '../DEPURATION/DATA/resultado_agrupado.xlsx'  # Cambia al nombre de tu archivo
df_victimas = pd.read_excel(archivo_victimas)

# Cargamos el segundo archivo (datos de viajeros y viajes) desde CSV
archivo_viajes = '../DEPURATION/DATA//movilidad_provincias_2023.csv'  # Cambia al nombre de tu archivo
df_viajes = pd.read_csv(archivo_viajes)

# Convertimos la columna de fecha 'day' en df_viajes a columnas de MES y DIA_MES
df_viajes['day'] = pd.to_datetime(df_viajes['day'])
df_viajes['MES'] = df_viajes['day'].dt.month
df_viajes['DIA_MES'] = df_viajes['day'].dt.day

# Unimos los datos de df_victimas con df_viajes para obtener METEO_ORIGEN y VICTIMAS_ORIGEN
resultado = pd.merge(
    df_viajes,
    df_victimas[['MES', 'DIA_MES', 'COD_PROVINCIA', 'TOTAL_VICTIMAS', 'CONDICION_METEO']],
    how='left',
    left_on=['MES', 'DIA_MES', 'provincia_origen'],
    right_on=['MES', 'DIA_MES', 'COD_PROVINCIA']
).rename(columns={'TOTAL_VICTIMAS': 'VICTIMAS_ORIGEN', 'CONDICION_METEO': 'METEO_ORIGEN'})

# Unimos nuevamente para obtener METEO_DESTINO y VICTIMAS_DESTINO
resultado = pd.merge(
    resultado,
    df_victimas[['MES', 'DIA_MES', 'COD_PROVINCIA', 'TOTAL_VICTIMAS', 'CONDICION_METEO']],
    how='left',
    left_on=['MES', 'DIA_MES', 'provincia_destino'],
    right_on=['MES', 'DIA_MES', 'COD_PROVINCIA'],
    suffixes=('', '_destino')
).rename(columns={'TOTAL_VICTIMAS': 'VICTIMAS_DESTINO', 'CONDICION_METEO': 'METEO_DESTINO'})

# Calculamos el total de víctimas sumando VICTIMAS_ORIGEN y VICTIMAS_DESTINO
resultado['TOTAL_VICTIMAS'] = resultado['VICTIMAS_ORIGEN'].fillna(0) + resultado['VICTIMAS_DESTINO'].fillna(0)

# Seleccionamos sólo las columnas relevantes para el resultado final
resultado = resultado[['viajeros', 'viajes', 'provincia_origen', 'provincia_origen_name',
                       'provincia_destino', 'provincia_destino_name', 'day',
                       'MES', 'DIA_MES', 'TOTAL_VICTIMAS', 'METEO_ORIGEN', 'METEO_DESTINO']]

# Guardamos el resultado en un archivo Excel
resultado.to_excel('datos_limpios.xlsx', index=False)

# Mostramos el resultado
resultado.head()
```

### datos_limpios.xlsx
After applying the processes outlined in the previous Notebook to our CSV files, we obtained the cleaned data, which is stored under the name `datos_limpios.xlsx`. This cleaned dataset will be used as the foundation for training our predictive model. It is essential to use this refined data, as it has undergone necessary transformations and filtering to ensure accuracy. By utilizing `datos_limpios.xlsx`, we can enhance the model's performance and reliability in predicting outcomes based on the our available features.

## MODEL dir
Inside the __MODEL__ directory, we have the following items:

- A Jupyter Notebook: This file contains the algorithm and code used to train our predictive model. It details the methodology, including data preprocessing, model selection, training procedures, and evaluation metrics.

- Accident predictions xlsx file: This file includes the estimated probabilities of accidents, which were generated after applying the model to a subset of our training data. This output is essential for understanding the model's predictive capabilities and assessing its performance.

### Predictive_model.ipynb
In this notebook, we implement the algorithm for our predictive model using the Random Forest methodology. This model aims to predict the probability of an accident based on various factors, including the provinces of origin and destination, meteorological conditions, and the specific day of the year. The code is organized into two main sections.

First, we focus on training the Random Forest model using the cleaned dataset. This section includes the implementation of the training process, where we apply the necessary algorithms to develop a robust predictive model. And also the displaying of the evaluating metrics that we obtain. The following code illustrates how we achieve this:

```python
mport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Carga de datos
archivo = "../DEPURATION/datos_limpios.xlsx"  # Cambia al nombre de tu archivo
df = pd.read_excel(archivo)

# Preprocesamiento

# Crear variable binaria para accidente: 1 si TOTAL_VICTIMAS > 0, de lo contrario 0
df['accidente_ocurrido'] = (df['TOTAL_VICTIMAS'] > 0).astype(int)

# Seleccionar variables de interés
features = ['MES', 'DIA_MES', 'METEO_ORIGEN', 'METEO_DESTINO', 'provincia_origen', 'provincia_destino']
X = df[features]
y = df['accidente_ocurrido']


# Codificar variables categóricas
X_encoded = pd.get_dummies(X, columns=['METEO_ORIGEN', 'METEO_DESTINO', 'provincia_origen', 'provincia_destino'], drop_first=True)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de accidente

# Evaluación del modelo
print("Exactitud:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
print("\nInforme de clasificación:\n", classification_report(y_test, y_pred))
```

In the second section, we display the results of the model’s predictions. Here, we also store these predictions in the `predicciones_accidentes.xlsx` file, allowing for further analysis and review of the predicted accident probabilities. The following code demonstrates this process:

```python
X_test_original = X.loc[X_test.index]  # Seleccionamos las filas originales correspondientes a X_test
resultado = X_test_original.copy()
resultado['PROBABILIDAD_DE_ACCIDENTE'] = y_pred_proba
resultado.to_excel('predicciones_accidentes.xlsx', index=False)
resultado.head()
```

### predicciones_accidentes.xlsx
After training the predictive model using our cleaned dataset, we generated accident predictions, which are stored in a file named `predicciones_accidentes.xlsx. This dataset correlates the origin and destination provinces with meteorological conditions, as well as the specific day and month of each prediction.By analyzing this data, we aim to understand how these variables interact and influence accident rates. This information could prove useful for identifying trends and implementing targeted safety measures to reduce accidents in the future.






