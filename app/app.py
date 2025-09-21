# CÓDIGO FINAL E DEFINITIVO para app.py
import streamlit as st
import pandas as pd
import joblib
from google.oauth2 import service_account
import gcsfs
import re

# --- Configuração da Página e Funções ---
st.set_page_config(
    page_title="Decision AI - Ranking de Candidatos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts_from_gcs(bucket_name):
    # ... (esta função está correta, não muda) ...
    try:
        with st.spinner("🔐 Autenticando com Google Cloud..."):
            creds_info = st.secrets["gcs_credentials"]
            scopes = [
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/devstorage.read_only',
            ]
            creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
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

# ✅ NOVA FUNÇÃO PARA OTIMIZAÇÃO DE VAGA
def gerar_recomendacoes_vaga(df_ranked, top_n_percent=0.20):
    """
    Analisa os melhores candidatos e sugere palavras-chave para otimizar a descrição da vaga.
    """
    # 1. Selecionar os melhores candidatos (top 20% por padrão)
    top_candidates = df_ranked.head(max(1, int(len(df_ranked) * top_n_percent)))
    
    # 2. Identificar as colunas de habilidades (skills)
    skill_columns = [col for col in top_candidates.columns if col.startswith('skill_')]
    
    # 3. Calcular a frequência de cada habilidade entre os melhores candidatos
    skill_counts = top_candidates[skill_columns].sum()
    # Filtrar habilidades que aparecem em pelo menos um candidato do top
    frequent_skills = skill_counts[skill_counts > 0].sort_values(ascending=False)
    
    if frequent_skills.empty:
        return "Nenhuma habilidade em comum encontrada entre os melhores candidatos para esta vaga.", []
        
    # 4. Extrair o texto original da descrição da vaga
    vaga_text = (df_ranked['principais_atividades'].iloc[0] + " " + df_ranked['competencia_tecnicas_e_comportamentais'].iloc[0]).lower()
    
    # 5. Identificar habilidades recomendadas que JÁ NÃO ESTÃO na descrição
    recomendacoes = []
    for skill_col_name, count in frequent_skills.items():
        # Limpar o nome da habilidade (ex: 'skill_gestao_de_projetos' -> 'gestão de projetos')
        skill_name = skill_col_name.replace('skill_', '').replace('_', ' ')
        # Usar regex para verificar se a habilidade já existe no texto
        if not re.search(r'\b' + re.escape(skill_name) + r'\b', vaga_text, re.IGNORECASE):
            recomendacoes.append(skill_name.capitalize())
            
    return frequent_skills.index.str.replace('skill_', '').str.replace('_', ' ').tolist(), recomendacoes

# --- Interface Principal ---
st.image("https://pos.fiap.com.br/wp-content/uploads/2022/07/pos-tech-fiap.svg", width=250)
st.title("🤖 Decision AI: Ferramenta de Otimização de Recrutamento")

BUCKET = "datathon-decision-ai-bolanos" 
model, model_columns, df_app = load_artifacts_from_gcs(BUCKET)

# --- Barra Lateral ---
# (sem alterações)

if model is None or df_app is None:
    st.error("A aplicação não pôde ser iniciada.")
    st.stop()

if 'titulo_vaga' in df_app.columns:
    lista_vagas = sorted(df_app['titulo_vaga'].astype(str).unique())
    vaga_selecionada = st.selectbox("**Selecione uma Vaga:**", options=lista_vagas, index=0)

    if vaga_selecionada:
        st.markdown("---")
        df_filtrado = df_app[df_app['titulo_vaga'] == vaga_selecionada].copy()
        
        # Pre-processamento para predição
        X_pred = pd.DataFrame(columns=model_columns)
        cols_comuns = [col for col in model_columns if col in df_filtrado.columns]
        X_pred = pd.concat([X_pred, df_filtrado[cols_comuns]]).fillna(0)
        
        pred_probs = model.predict_proba(X_pred[model_columns])[:, 1]
        df_resultados = df_filtrado.copy()
        df_resultados['match_score'] = (pred_probs * 100)
        df_resultados = df_resultados.sort_values(by='match_score', ascending=False)
        
        # --- ABAS DE NAVEGAÇÃO ---
        tab_ranking, tab_otimizacao = st.tabs(["🏆 Ranking de Candidatos", "✨ Otimizar Vaga com IA"])

        with tab_ranking:
            # (Código do ranking como antes)
            st.header(f"Ranking para: {vaga_selecionada}")
            col1, col2, col3 = st.columns(3)
            # ... (métricas e dataframe) ...

        with tab_otimizacao:
            st.header("🤖 Assistente de Otimização de Vaga")
            st.markdown("A IA analisou os CVs dos candidatos com maior *match score* para esta vaga e sugere melhorias para a sua descrição.")
            
            # Gerar e exibir as recomendações
            common_skills, recommendations = gerar_recomendacoes_vaga(df_resultados)

            st.markdown("---")
            st.subheader("Palavras-chave a serem adicionadas")
            if recommendations:
                st.success("Para atrair candidatos mais qualificados, considere adicionar as seguintes habilidades à descrição da vaga:")
                # Exibir em colunas para melhor visualização
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    cols[i % 3].markdown(f"- **{rec}**")
            else:
                st.info("A descrição desta vaga já parece bem alinhada com as habilidades dos melhores candidatos. Bom trabalho!")

            with st.expander("Ver lógica da IA"):
                st.markdown("#### Como as recomendações foram geradas?")
                st.write(f"1. A IA selecionou os **{max(1, int(len(df_resultados) * 0.20))} melhores candidatos** (top 20%) com base no *match score*.")
                st.write("2. Analisou os CVs deste grupo e identificou as habilidades mais frequentes.")
                st.code(f"Habilidades mais comuns no top 20%: {', '.join(common_skills[:5])}...")
                st.write("3. Comparou essas habilidades com o texto da descrição da vaga atual.")
                st.write("4. As sugestões acima são as habilidades frequentes nos melhores candidatos que **não** foram encontradas na sua descrição.")

else:
    st.warning("A coluna 'titulo_vaga' não foi encontrada no dataset.")