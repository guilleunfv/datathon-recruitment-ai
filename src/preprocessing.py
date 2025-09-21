# src/preprocessing.py
import pandas as pd

def carregar_e_unir_dados(gcs_bucket_path):
    """Carrega os dados brutos do GCS e os une em um DataFrame mestre."""
    print("Carregando dados brutos do GCS...")
    df_applicants = pd.read_json(f"{gcs_bucket_path}/data/applicants.json", orient='index')
    df_vagas = pd.read_json(f"{gcs_bucket_path}/data/vagas.json", orient='index')
    df_prospects = pd.read_json(f"{gcs_bucket_path}/data/prospects.json", orient='index')

    print("Aplanando DataFrames...")
    # Aplanar Applicants
    # (código de aplanamento de applicants que você criou)
    # ...
    # Aplanar Vagas
    # (código de aplanamento de vagas que você criou)
    # ...
    # Aplanar Prospects
    # (código de aplanamento de prospects que você criou)
    # ...

    print("Unindo DataFrames...")
    # Unir os DataFrames
    # (código de união para criar df_mestre que você criou)
    # ...
    
    print("Dados processados com sucesso!")
    return df_mestre