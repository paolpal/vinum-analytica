# Importare le librerie necessarie
import pandas as pd
from unidecode import unidecode
import re
# Caricare il dataset
data_path = './data/raw/winemag-data-130k-v2.csv'  # Modificare il percorso se necessario
df = pd.read_csv(data_path, index_col=0)

# rimuovi le recensioni senza prezzo
df = df.dropna(subset=['price'])

# Salvare il dataset pulito
clean_data_path = './data/raw/clean_data_regression.csv'  # Modificare il percorso se necessario
df.to_csv(clean_data_path, index=False)