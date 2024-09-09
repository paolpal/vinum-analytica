# Importare le librerie necessarie
import pandas as pd
from unidecode import unidecode
import re
# Caricare il dataset
data_path = './data/raw/winemag-data-130k-v2.csv'  # Modificare il percorso se necessario
df = pd.read_csv(data_path, index_col=0)

df = df[~df['variety'].str.contains('Blend', case=False, na=False)]  # Rimuovere i blend
variety_counts = df['variety'].value_counts()

support_threshold = 0.006*len(df)  # Set the support threshold as desired

selected_varieties = variety_counts[variety_counts >= support_threshold]

filtered_df = df[df['variety'].isin(selected_varieties.index)]
df = filtered_df


df['variety_contamination'] = df.apply(lambda row: unidecode(row['variety']).lower() in unidecode(row['description']).lower(), axis=1)
contamination_count = df['variety_contamination'].sum()
print(f'Numero di campioni contaminati: {contamination_count}/{len(df)}')

def remove_variety_from_description(row):
    variety_words = unidecode(row['variety']).lower().split(' ')  # Rimuove accenti e trasforma in minuscolo
    description = row['description']  # Usa la descrizione della riga
    clean_description = description  # Copia della descrizione originale
    
    # Itera su ogni parola della varietà e rimuovila dalla descrizione
    for word in variety_words:
        # Crea una regex per trovare la parola come parola intera, ignorando il case
        regex = r'\b' + re.escape(unidecode(word).lower()) + r'(s|es|ies)?\b'
        clean_description = re.sub(regex, '', unidecode(clean_description), flags=re.IGNORECASE)

    return clean_description.strip()  # Rimuovi eventuali spazi aggiuntivi


# Applicare la funzione a ogni riga del dataframe
df['description'] = df.apply(remove_variety_from_description, axis=1)

df['variety_contamination'] = df.apply(lambda row: unidecode(row['variety']).lower() in unidecode(row['description']).lower(), axis=1)
contamination_count = df['variety_contamination'].sum()
print(f'Numero di campioni contaminati: {contamination_count}/{len(df)}')

# Print one of the contaminated description and variety
contaminated_samples = df[df['variety_contamination'] == True]
# select only chardonnaies
contaminated_samples = contaminated_samples[contaminated_samples['variety'] == 'Malbec']
for index, row in contaminated_samples.iterrows():
    contaminated_description = row['description']
    contaminated_variety = row['variety']
    print(f'Contaminated Description: {contaminated_description}')
    print(f'Contaminated Variety: {contaminated_variety}')

contaminated_varieties = df[df['variety_contamination'] == True]['variety'].unique()
print(f'Varieties with contamination: {contaminated_varieties}')

df = df.drop(columns=['variety_contamination'])

# Salvare il dataset pulito
clean_data_path = './data/raw/clean_data.csv'  # Modificare il percorso se necessario
df.to_csv(clean_data_path, index=False)