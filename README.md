# PM-Telef-101C
Repository including the mobility data provided by telefonica, the siniestrality data, and meteorologic data that we are going to use , the code that we will apply to clean the data and the algorithm to train our predictive model.

To install the materials, create a virtual environment and install the dependencies.
On Mac/Linux this will be:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## DEPURATION dir
Inside depuration directory we have the DATA directory, the xlsx file including the depurated data and finally the MergingInfo script that is the one that we use in order to obtain this clean data. Notice that the script is included just to have the materials in order to perform  the merging in diferent data, but we have already applied this script to our dirty data corresponding to 2023 mobility and accidents included in DATA folder.

### DATA dir
In DATA directory we have two main csv files and one auxiliary file that is used to obtain the clean data. In `ACCIDENTES_METEO-TABLA.csv` we include
the datat relating the accidents occured in a province in each day of the year depending on the meteorological condition. In `movilidad_provincias_2023.csv` we have the information about the number of displacements and the number of travellers from one province to another each day of the year.

### MergingInfo.ipynb
In this notebook we apply some changes to the two previous csv files to depurate our data and be able to start training our predictive model. The code is divided in two main parts. Firstly we clean the `ACCIDENTES_METEO-TABLA.csv` file using the following code:

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
