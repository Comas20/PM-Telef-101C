# PM-Telef-101C
Repository including the mobility data provided by telefonica, the siniestrality data, and meteorologic data that we are going to use , the code that we will apply to clean the data and the algorithm to train our predictive model.

To install the materials, create a virtual environment and install the dependencies.
On Mac/Linux this will be:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## 1.1 DEPURATION dir
__IMPORTANT: *to access the data inside the DEPURATION directory, you must request permission from the Project Manager (alvaro.monterrubio01@estudiant.upf.edu). Access to this data is restricted due to protections in place by Telefónica and other collaborating entities to ensure data security and confidentiality.*__

Inside the __DEPURATION__ directory, we have the following items:

- DATA Directory: This directory contains our original datasets, which include raw data regarding mobility and accidents from 2023. The data within this folder has not been modified and is kept intact for reference and future use.

- Cleaned Data xlsx File: This Excel file contains the cleaned and processed data, derived from the original datasets in the DATA directory. The cleaning process involves removing inaccuracies, handling missing values, and restructuring the data to make it suitable for analysis.
  
- Jupyter Notebook: This notebook provides a detailed overview of the data cleaning and merging process we employed to obtain the cleaned dataset. It includes the code used, along with explanations of each step taken during the data preparation phase. The notebook serves as a valuable resource for anyone looking to replicate our methodology or apply similar processes to different datasets.

Please note that the Jupyter Notebook is included as documentation of our work and also as a resource for merging different datasets in the future. We have already applied this process to our raw data related to mobility and accidents from 2023, users can leverage the notebook to adapt the cleaning and merging techniques to their own datasets as needed.

### 1.1.2 DATA dir
In the __DATA directory__, we have two main CSV files and one auxiliary file that is essential for obtaining the cleaned data.

- `ACCIDENTES_METEO-TABLA.csv`: This file contains detailed data regarding accidents that occurred in various provinces throughout the year. For each day, it contains information about the number of accidents in each province, with the corresponding meteorological conditions on that day. This dataset is crucial for understanding how weather influences accident rates.
  
- `movilidad_provincias_2023.csv`: This file provides comprehensive information on the number of displacements and travelers moving from one province to another each day of the year. It includes metrics such as the total number of trips made and the origin and destination provinces. This dataset is necessary for analyzing mobility patterns and understanding how travel behavior may correlate with accident occurrences.
  
- Auxiliary File: An additional auxiliary file is included in this directory to facilitate the cleaning and merging processes. This file is not directly involved in the final analysis, It contains a cleaned version of the `ACCIDENTES_METEO-TABLA.csv` file that will be use to merge it with the `movilidad_provincias_2023.csv` file.
  
Together, these datasets form the backbone of our analysis, they providing the necessary information to examine the relationships between mobility, accidents, and weather conditions.

### 1.1.3 MergingInfo.ipynb
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

## 1.2 MODEL dir
Inside the __MODEL__ directory, we have the following items:

- A Jupyter Notebook with the first model version: This file contains the algorithm and code used to train our predictive model. It details the methodology, including data preprocessing, model selection, training procedures, and evaluation metrics.

- Accident predictions xlsx file for small test set: This file includes the estimated probabilities of accidents, which were generated after applying the model to a subset of our training data. This output is essential for understanding the model's predictive capabilities and assessing its performance.

- A Jupyter Notebook with the updated model version: This file contains the algorithm and code used to save the weights obtained after training our predictive model with the same configuration as the initial version. Additionally, it includes code to predict accident probabilities for the entire cleaned dataset. The notebook provides a detailed methodology, including data preprocessing steps, model selection rationale, training procedures, and evaluation metrics.

- Accident predictions xlsx file for the whole dataset: This file includes the estimated probabilities of accidents, which were generated after applying the model to the whole cleaned dataset. This output is essential for understanding the model's predictive capabilities and assessing its performance.

- Pre trained weights: This text file provides a link to a Google Drive folder containing the model weights saved after the initial training. These pre-trained weights enable the model to be applied to new datasets for prediction purposes without needing to retrain from scratch, streamlining future analysis and ensuring consistency with the original model configuration.

### 1.2.1 Predictive_model.ipynb
In this notebook, we implement the algorithm for our predictive model using the Random Forest methodology. This model aims to predict the probability of an accident based on various factors, including the provinces of origin and destination, meteorological conditions, and the specific day of the year. The code is organized into two main sections.

First, we focus on training the Random Forest model using the cleaned dataset. This section includes the implementation of the training process, where we apply the necessary algorithms to develop a robust predictive model. And also the displaying of the evaluating metrics that we obtain. The following code illustrates how we achieve this:

```python
import pandas as pd
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

To evaluate the performance of the model after this initial training configuration, we have decided to present several accuracy metrics, detailed below. These metrics will provide insights into the model’s predictive quality and help identify areas for potential improvement in future iterations.

```
Exactitud: 0.9734291094218565
AUC-ROC: 0.9965846036793008

Informe de clasificación:
               precision    recall  f1-score   support

           0       0.98      0.98      0.98     98563
           1       0.94      0.95      0.94     30488

    accuracy                           0.97    129051
   macro avg       0.96      0.96      0.96    129051
weighted avg       0.97      0.97      0.97    129051

```

### 1.2.2 predicciones_accidentes.xlsx
After training the predictive model using our cleaned dataset, we generated accident predictions, which are stored in a file named `predicciones_accidentes.xlsx`. This dataset correlates the origin and destination provinces with meteorological conditions, as well as the specific day and month of each prediction.By analyzing this data, we aim to understand how these variables interact and influence accident rates. 

### 1.2.3 Predictive_model_V2.ipynb

After the initial model execution and completion, we implemented code to save the model weights for use in future training with other datasets. These weights have been uploaded to a secure Google Drive directory. To access this confidential file, you can request access via the link provided in the `pre_trained_model_weights.txt` file. The code used to store the model weights is shown below:

```python
import joblib
joblib.dump(model, 'modelo_random_forest_weights.pkl')
print("Modelo guardado como 'modelo_random_forest.pkl'")
```

Oce the wights are stored, we enhanced the previous version of the predictive model script to calculate the accident risk for all entries in `datos_limpios.xlsx`. To generate these predictions for the complete clean dataset, we first loaded the stored weights, then ran the prediction process on the entire dataset. The code used to load the weights and generate the predictions is shown below:

```python
model_cargado = joblib.load('../MODEL/modelo_random_forest_weights.pkl')
archivo = "../DEPURATION/datos_limpios.xlsx"  # Cambia al nombre de tu archivo

df = pd.read_excel(archivo)
df['accidente_ocurrido'] = (df['TOTAL_VICTIMAS'] > 0).astype(int)

X_v2 = df[features]
y_v2 = df['accidente_ocurrido']

X_v2_encoded = pd.get_dummies(X_v2, columns=['METEO_ORIGEN', 'METEO_DESTINO', 'provincia_origen', 'provincia_destino'], drop_first=True)
y_pred_nuevo = model_cargado.predict(X_v2_encoded)
y_pred_proba_nuevo = model_cargado.predict_proba(X_v2_encoded)[:, 1]

print("Exactitud:", accuracy_score(y_v2, y_pred_nuevo))
print("AUC-ROC:", roc_auc_score(y_v2, y_pred_proba_nuevo))
print("\nInforme de clasificación:\n", classification_report(y_v2, y_pred_nuevo))
```

After generating the predictions, we displayed the model’s results and saved them to the file `final_predicciones_accidentes.xlsx` for further analysis and review of the predicted accident probabilities. The code below demonstrates this process:

```python
X_test_total = X_v2.loc[X_v2_encoded.index]  # Seleccionamos las filas originales correspondientes a X_test
resultado = X_test_total.copy()
resultado['PROBABILIDAD_DE_ACCIDENTE'] = y_pred_proba_nuevo
resultado.to_excel('final_predicciones_accidentes.xlsx', index=False)
resultado.head()
```

Again, to evaluate the performance of the model after this initial training configuration, we have decided to present several accuracy metrics, detailed below. These metrics will provide insights into the model’s predictive quality and help identify areas for potential improvement in future iterations.

```
Exactitud: 0.9946858136485787
AUC-ROC: 0.999633461635886

Informe de clasificación:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    493039
           1       0.99      0.99      0.99    152215

    accuracy                           0.99    645254
   macro avg       0.99      0.99      0.99    645254
weighted avg       0.99      0.99      0.99    645254
```

### 1.2.4 final_predicciones_accidentes.xlsx

After loading the weights obtained from the initial training, we generated accident predictions for the entire cleaned dataset, storing the results in `final_predicciones_accidentes.xlsx`. This dataset includes information linking origin and destination provinces, meteorological conditions, and specific day and month details for each prediction. By analyzing these variables, we aim to understand their interaction and influence on accident rates. This insight could help identify patterns and support targeted safety measures to reduce accidents in the future.

## 1.3 DASHBOARD dir

In the __DASHBOARD Directory__ it is stored the Power BI interactive dashboard of accidents prediction.

- dashbord txt file: This file contains the link to the implemented interactive dashboard.

### 1.3.1 Dashboard structure

With the tool Power BI as a main developer, an interactive dashboard with data analyitics has been created in order to identifying trends and implement targeted safety measures to reduce accidents in the future.  It is very important to select the most appropiate charts and diagrams that show the most relevant information. 

This is why we have selected:

- **Interactive Map**:  
  - Purpose: To show the geographical distribution of accidents across the provinces.  
  - Reason for Selection: This chart provides a clear spatial representation of areas with higher accident risks, enabling quick identification of high-risk zones for targeted safety interventions.

- **Stacked Bar Chart**:  
  - Purpose: To display the proportion of accident probabilities across different provinces.  
  - Reason for Selection: This visualization allows comparison of accident risk across provinces, helping to prioritize regions requiring attention.

- **Pie Chart**:  
  - Purpose: To break down the proportion of accidents by weather conditions (e.g., rainy, sunny, foggy).  
  - Reason for Selection: Highlights the influence of weather on accident rates, useful for designing weather-specific safety measures.

In addition to these charts, the dashboard includes:

- **Filters**:  
  - Purpose: To dynamically refine the displayed data based on the province, date and weather conditions. 
  - Reason for Selection: Filters enhance user interaction, allowing stakeholders to focus on areas of particular interest.

- **Cards**:  
  - Purpose: To display key metrics such as number of accidents, number of victims, accident rate (`nº accidents/nº trips`).  
  - Reason for Selection: Cards provide a quick summary of critical data points.


### 1.3.2 Insertion of data

Once the structure of the dashboard is completed showing all charts and diagrams of most interest, it is time to introduce the dataset `final_predicciones_accidentes.xlsx`.

### 1.3.3 Dashboard evaluation
Evaluating the dashboard is a crucial step to ensure its accuracy, functionality, and effectiveness as an analytical tool. This dashboard, developed in Power BI, is designed to provide an interactive and visual representation of accident data, enabling users to identify patterns and make informed decisions. Below are the details of the evaluation for the dashboard's various components:

**1.3.3.1 Data Accuracy Evaluation**
Data Source: The dashboard uses the final_predicciones_accidentes.xlsx file as its data source, containing detailed information on accidents, provinces, weather conditions, and daily statistics.
Data Verification: Key metrics such as the total number of accidents, accident rate, and total number of victims were cross-checked against the source data to ensure the integrity of the information presented.

**1.3.3.2 Functionality Testing**
Accident Probability by Province Bar Chart: This bar chart was tested to confirm that it updates correctly and displays provinces with higher accident probabilities, highlighting critical areas like Alicante, Badajoz, and the Balearic Islands.
Interactive Geographic Map: The map was tested for responsiveness, ensuring it allows users to zoom in, click on different provinces, and view real-time data. The shading intensity varies accurately according to the number of registered accidents.
Weather Condition Pie Chart: The pie chart was verified to ensure it accurately displays the distribution of accidents by weather conditions, such as rain, sun, and fog.
General Metrics Cards: Cards displaying key figures, like the total number of accidents, accident rate, and number of victims, were checked to confirm proper display and automatic updates when filters are applied.

**1.3.3.3 Visual Consistency**
Chart Design: All visual elements, including the map and charts, were evaluated for consistency in color scheme, formatting, and layout, facilitating clear data interpretation.
Labels and Legends: It was verified that all charts have clear labels, titles, and legends that accurately describe the data being presented, making the visualization understandable for users.
Information Layout: The dashboard layout was checked to ensure all elements are logically and accessibly aligned, making it easy for users to find relevant information.

**1.3.3.4 Performance Review**
Load Times: The dashboard was tested to evaluate load times and the responsiveness of interactive elements when filters and selections are applied. It was found to perform efficiently with minimal load times.
Responsiveness: The dashboard’s adaptability to different devices and screen sizes was reviewed, ensuring it remains functional and visually consistent on desktops, laptops, and tablets.

**1.3.3.5 Insight and Actionability Analysis**
Decision-Making Support: The evaluation confirmed that the dashboard provides actionable insights that help users identify high-risk areas, assess the impact of weather conditions, and observe regional accident patterns.
Clarity of Data Presentation: The dashboard's visualizations, such as the accident probability by province bar chart and the weather conditions pie chart, were reviewed to ensure the data is presented in a clear and understandable manner, facilitating quick identification of areas that need attention.

### 1.3.4 Final Statement

We are excited to announce that this project has been successfully completed, fulfilling all the objectives and delivering valuable results. As part of this work, we have developed a highly accurate predictive model and an interactive dashboard designed to enhance decision-making and user engagement.

The predictive model was created with a focus on precision, scalability, and reliability. It incorporates advanced methodologies to ensure robust forecasting capabilities that can adapt to diverse scenarios. Every stage of the model's development, from data preprocessing to validation, was executed with meticulous attention to detail to guarantee the best possible performance.

In addition to the predictive model, we have created an interactive dashboard tailored for an intuitive and seamless user experience. This dashboard empowers users to explore data and predictions dynamically, offering a wide range of features. Its design emphasizes usability and flexibility, making it a valuable tool for uncovering insights and guiding strategic decisions.

This project reflects our commitment to combining technical rigor with practical utility. We are confident that the deliverables will provide meaningful impact and invite you to explore the repository to review the implementation, test the tools, and share your feedback. Thank you for your interest and support throughout this journey!



