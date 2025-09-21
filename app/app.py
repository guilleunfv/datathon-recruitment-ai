# CÓDIGO FINAL E DEFINITIVO para app.py
import streamlit as st
import pandas as pd
import joblib
from google.oauth2 import service_account
import gcsfs

st.set_page_config(
    page_title="Decision AI - Ranking de Candidatos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts_from_gcs(bucket_name):
    try:
        with st.spinner("🔐 Autenticando com Google Cloud..."):
            creds_info = st.secrets["gcs_credentials"]
            
            # CORREÇÃO DEFINITIVA: Definimos o 'scope' (permissão) explicitamente.
            # Isto resolve o erro 'invalid_scope' em ambientes não interativos.
            scopes = [
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/devstorage.read_only',
            ]
            creds = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=scopes
            )
            gcs = gcsfs.GCSFileSystem(project=creds_info['project_id'], token=creds)

        caminho_modelo_gcs = f"gs://{bucket_name}/models/recruitment_model.joblib"
        caminho_colunas_gcs = f"gs://{bucket_name}/models/model_columns.joblib"
        caminho_dataset_gcs = f"gs://{bucket_name}/data/df_mestre_preprocessado.parquet"

        with st.spinner("📦 Carregando modelo, colunas e dataset do GCS..."):
            with gcs.open(caminho_modelo_gcs, 'rb') as f:
                model = joblib.load(f)
            with gcs.open(caminho_colunas_gcs, 'rb') as f:
                model_columns = joblib.load(f)
            df_app = pd.read_parquet(caminho_dataset_gcs, filesystem=gcs)
        
        return model, model_columns, df_app
    except Exception as e:
        st.error(f"❌ Erro fatal ao carregar artefatos do GCS: {e}")
        return None, None, None

st.image("https://pos.fiap.com.br/wp-content/uploads/2022/07/pos-tech-fiap.svg", width=250)
st.title("🤖 Decision AI: Ranking de Candidatos")
st.markdown("Uma ferramenta de IA para ajudar recrutadores a encontrar os melhores talentos de forma eficiente.")

BUCKET = "datathon-decision-ai-bolanos" 
model, model_columns, df_app = load_artifacts_from_gcs(BUCKET)

with st.sidebar:
    st.header("Datathon")
    st.write("**Integrantes:**")
    st.info("""
    - Rosicléia Cavalcante Mota
    - Guillermo J. Camahuali Privat
    - Kelly Priscilla Matos Campos
    """)
    st.markdown("---")
    st.header("Sobre o Modelo")
    st.markdown("""
    - **Algoritmo:** XGBoost
    - **Exatidão (Accuracy):** 82%
    - **Recall (Contratados):** 51%
    """)
    st.markdown("Apoiado por Streamlit e Google Cloud.")

if model is None or df_app is None:
    st.error("A aplicação não pôde ser iniciada. Verifique os erros de carregamento acima.")
    st.stop()

st.success("✅ Modelo e dados carregados! Selecione uma vaga para começar.")

if 'titulo_vaga' in df_app.columns:
    lista_vagas = sorted(df_app['titulo_vaga'].astype(str).unique())
    vaga_selecionada = st.selectbox("**Selecione uma Vaga:**", options=lista_vagas, index=0)

    if vaga_selecionada:
        st.markdown("---")
        df_filtrado = df_app[df_app['titulo_vaga'] == vaga_selecionada].copy()
        
        cols_comuns = [col for col in model_columns if col in df_filtrado.columns]
        X_pred = pd.DataFrame(columns=model_columns)
        X_pred = pd.concat([X_pred, df_filtrado[cols_comuns]]).fillna(0)

        pred_probs = model.predict_proba(X_pred[model_columns])[:, 1]
        df_filtrado['match_score'] = (pred_probs * 100)
        df_resultados = df_filtrado.sort_values(by='match_score', ascending=False)
        
        st.header(f"🏆 Ranking de Candidatos para: {vaga_selecionada}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Candidatos", len(df_resultados))
        col2.metric("Melhor Match Score", f"{df_resultados['match_score'].max():.2f}%")
        col3.metric("Match Score Médio", f"{df_resultados['match_score'].mean():.2f}%")
        
        cols_ranking = ['nome', 'match_score', 'similitude_cv_vaga', 'anos_experiencia', 'email']
        cols_ranking_existentes = [col for col in cols_ranking if col in df_resultados.columns]
        
        if cols_ranking_existentes:
            st.dataframe(
                df_resultados[cols_ranking_existentes],
                use_container_width=True,
                column_config={
                    "nome": st.column_config.TextColumn("Nome do Candidato", width="large"),
                    "match_score": st.column_config.ProgressColumn("Match Score (%)", format="%.2f%%", min_value=0, max_value=100),
                    "similitude_cv_vaga": st.column_config.NumberColumn("Similitude CV", format="%.2f"),
                    "anos_experiencia": st.column_config.NumberColumn("Anos Exp.", format="%d anos"),
                    "email": st.column_config.TextColumn("E-mail")
                },
                hide_index=True
            )
else:
    st.warning("A coluna 'titulo_vaga' não foi encontrada no dataset.")