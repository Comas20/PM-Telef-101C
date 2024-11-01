# PM-Telef-101C
Repository including the mobility data provided by telefonica, the siniestrality data, and meteorologic data that we are going to use , the code that we will apply to clean the data and the algorithm to train our predictive model.

To install the materials, create a virtual environment and install the dependencies.
On Mac/Linux this will be:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## DEPURATION Directory
Inside depuration directory we have the DATA directory, the xlsx file including the depurated data and finally the MergingInfo script that is the one that we use in order to obtain this clean data. Notice that the script is included just to have the materials in order to perform  the merging in diferent data, but we have already applied this script to our dirty data corresponding to 2023 mobility and accidents included in DATA folder.

### DATA Directory
In DATA directory we have two main csv files and one auxiliary file that is used to obtain the clean data. In ACCIDENTES_METEO-TABLA.csv we include
