# src/train.py
from preprocessing import carregar_e_unir_dados
from feature_engineering import criar_features
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

BUCKET_PATH = "gs://datathon-decision-ai-bolanos"

# 1. Carga e Pré-processamento
df_mestre = carregar_e_unir_dados(BUCKET_PATH)

# 2. Engenharia de Features
df_com_features = criar_features(df_mestre)
df_com_features.to_parquet(f'{BUCKET_PATH}/data/df_mestre_preprocessado.parquet')
print("DataFrame com features salvo em GCS.")

# 3. Preparação Final para o Modelo
# (código da Célula 9 do notebook para criar o df_modelo)
# ...

# 4. Divisão dos Dados
y = df_modelo['contratado']
X = df_modelo.drop(columns='contratado')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Treinamento
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False)
model.fit(X_train, y_train)
print("Modelo treinado com sucesso.")

# 6. Salvar Artefatos
joblib.dump(model, '../models/recruitment_model.joblib')
joblib.dump(X_train.columns, '../models/model_columns.joblib')
print("Modelo e colunas salvos na pasta /models.")