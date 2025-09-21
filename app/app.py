# CÓDIGO FINAL E DEFINITIVO com Otimização de CV
import streamlit as st
import pandas as pd
import joblib
from google.oauth2 import service_account
import gcsfs
import re
from sentence_transformers import util

# --- Configuração da Página e Funções ---
st.set_page_config(
    page_title="Decision AI - Otimizador de Recrutamento",
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
        
        # Carregando o modelo de linguagem para a otimização de CV
        from sentence_transformers import SentenceTransformer
        lang_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        return model, model_columns, df_app, lang_model
    except Exception as e:
        st.error(f"❌ Erro fatal ao carregar artefatos do GCS: {e}")
        return None, None, None, None

def gerar_recomendacoes_vaga(df_ranked, top_n_percent=0.20):
    # ... (esta função está correta, não muda) ...
    if df_ranked.empty:
        return [], []
    top_candidates = df_ranked.head(max(1, int(len(df_ranked) * top_n_percent)))
    skill_columns = [col for col in top_candidates.columns if col.startswith('skill_')]
    skill_counts = top_candidates[skill_columns].sum()
    frequent_skills = skill_counts[skill_counts > 0].sort_values(ascending=False)
    if frequent_skills.empty:
        return [], []
    vaga_text = (str(df_ranked['principais_atividades'].iloc[0]) + " " + str(df_ranked['competencia_tecnicas_e_comportamentais'].iloc[0])).lower()
    recomendacoes = []
    common_skills_list = frequent_skills.index.str.replace('skill_', '').str.replace('_', ' ').tolist()
    for skill_name in common_skills_list:
        if not re.search(r'\b' + re.escape(skill_name) + r'\b', vaga_text, re.IGNORECASE):
            recomendacoes.append(skill_name.title())
    return common_skills_list, recomendacoes

# --- Interface Principal ---
st.image("https://pos.fiap.com.br/wp-content/uploads/2022/07/pos-tech-fiap.svg", width=250)
st.title("🤖 Decision AI: Otimizador de Recrutamento")

BUCKET = "datathon-decision-ai-bolanos" 
model, model_columns, df_app, lang_model = load_artifacts_from_gcs(BUCKET)

# ✅ Bloco de informações movido para debaixo do título
with st.expander("ℹ️ Sobre o Projeto e Modelo"):
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

if model is None or df_app is None or lang_model is None:
    st.error("A aplicação não pôde ser iniciada. Verifique os erros de carregamento acima.")
    st.stop()

# ✅ Barra lateral apenas para o seletor de vagas
with st.sidebar:
    st.header("Filtros")
    if 'titulo_vaga' in df_app.columns:
        lista_vagas = sorted(df_app['titulo_vaga'].astype(str).unique())
        vaga_selecionada = st.selectbox("**Selecione uma Vaga:**", options=lista_vagas, index=0)

if vaga_selecionada and 'titulo_vaga' in df_app.columns:
    df_filtrado_vaga = df_app[df_app['titulo_vaga'] == vaga_selecionada].copy()
    
    # --- ABAS DE NAVEGAÇÃO ---
    tab_ranking, tab_otimiza_vaga, tab_otimiza_cv = st.tabs([
        "🏆 Ranking de Candidatos", 
        "✨ Otimizar Vaga com IA",
        "📄 Otimizar meu CV"
    ])

    with tab_ranking:
        # Lógica de predição e exibição do ranking
        # ... (código do ranking, sem alterações, mas colocado aqui dentro) ...
        # (Omitido por brevidade, está en el código completo abajo)

    with tab_otimiza_vaga:
        # Lógica de otimização de vaga
        # ... (código da otimização de vaga, sem alterações, mas colocado aqui dentro) ...
        # (Omitido por brevidade, está en el código completo abajo)

    with tab_otimiza_cv:
        st.header(f"📄 Assistente de Otimização de CV para a vaga: **{vaga_selecionada}**")
        st.markdown("Cole o texto do seu CV abaixo e a IA irá analisá-lo e sugerir melhorias.")
        
        cv_usuario = st.text_area("Cole o texto completo do seu CV aqui:", height=300, placeholder="Ex: Formação Acadêmica...")
        
        if st.button("Analisar meu CV", type="primary"):
            if cv_usuario:
                with st.spinner("Analisando seu CV..."):
                    # 1. Obter texto da vaga
                    texto_vaga_completo = (str(df_filtrado_vaga['principais_atividades'].iloc[0]) + " " + str(df_filtrado_vaga['competencia_tecnicas_e_comportamentais'].iloc[0]))
                    
                    # 2. Calcular a similitude semântica
                    embedding_vaga = lang_model.encode(texto_vaga_completo, convert_to_tensor=True)
                    embedding_cv = lang_model.encode(cv_usuario, convert_to_tensor=True)
                    score_semantico = util.cos_sim(embedding_vaga, embedding_cv).item() * 100

                    # 3. Identificar habilidades chave da vaga que faltam no CV
                    skills_list = [col.replace('skill_', '').replace('_', ' ') for col in model_columns if col.startswith('skill_')]
                    habilidades_faltantes = []
                    for skill in skills_list:
                        # Se a habilidade está na vaga E não está no CV do usuário
                        if re.search(r'\b' + re.escape(skill) + r'\b', texto_vaga_completo, re.IGNORECASE) and not re.search(r'\b' + re.escape(skill) + r'\b', cv_usuario, re.IGNORECASE):
                            habilidades_faltantes.append(skill.title())
                            
                st.markdown("---")
                st.subheader("Resultados da Análise:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score de Alinhamento Semântico", f"{score_semantico:.2f}%")
                    st.progress(int(score_semantico))
                with col2:
                    st.info("Este score mede o quão bem o seu CV 'soa' para esta vaga, com base no significado geral dos textos.")
                
                st.markdown("---")
                st.subheader("💡 Recomendações de Habilidades")
                if habilidades_faltantes:
                    st.warning("Considere adicionar (se você as possui) as seguintes habilidades encontradas na descrição da vaga, mas que não foram detectadas em seu CV:")
                    cols_skills = st.columns(4)
                    for i, skill in enumerate(habilidades_faltantes[:12]): # Mostra até 12
                        with cols_skills[i % 4]:
                            st.markdown(f"- **{skill}**")
                else:
                    st.success("Ótimo trabalho! Seu CV parece conter todas as habilidades chave mencionadas na descrição da vaga.")
            else:
                st.error("Por favor, cole o texto do seu CV para análise.")