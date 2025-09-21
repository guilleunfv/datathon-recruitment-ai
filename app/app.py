# CÓDIGO COMPLETO E CORRIGIDO para app.py
import streamlit as st
import pandas as pd
import joblib
from google.oauth2 import service_account
import gcsfs
import re

# --- Configuração da Página e Funções ---
st.set_page_config(
    page_title="Decision AI - Otimizador de Recrutamento",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts_from_gcs(bucket_name):
    """
    Função para baixar e carregar o modelo, as colunas e o dataset do GCS.
    """
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
    """
    Analisa os melhores candidatos e sugere palavras-chave para otimizar a descrição da vaga.
    """
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

with st.sidebar:
    st.header("Filtros")
    if 'titulo_vaga' in df_app.columns:
        lista_vagas = sorted(df_app['titulo_vaga'].astype(str).unique())
        vaga_selecionada = st.selectbox("**Selecione uma Vaga:**", options=lista_vagas, index=0)

if vaga_selecionada and 'titulo_vaga' in df_app.columns:
    df_filtrado_vaga = df_app[df_app['titulo_vaga'] == vaga_selecionada].copy()
    
    tab_ranking, tab_otimiza_vaga, tab_otimiza_cv = st.tabs([
        "🏆 Ranking de Candidatos", 
        "✨ Otimizar Vaga com IA",
        "📄 Otimizar meu CV"
    ])

    with tab_ranking:
        st.header(f"Ranking para: {vaga_selecionada}")
        
        # Bloco de pré-processamento para a predição
        colunas_categoricas = ['nivel profissional', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol', 'vaga_sap', 'tipo_contratacao']
        colunas_categoricas_existentes = [col for col in colunas_categoricas if col in df_filtrado_vaga.columns]
        df_pred_processed = pd.get_dummies(df_filtrado_vaga, columns=colunas_categoricas_existentes, prefix=colunas_categoricas_existentes)
        for col in model_columns:
            if col in df_pred_processed.columns and df_pred_processed[col].dtype == 'object':
                df_pred_processed[col] = pd.to_numeric(df_pred_processed[col], errors='coerce').fillna(0)
        X_pred = df_pred_processed.reindex(columns=model_columns, fill_value=0)

        # Predição
        pred_probs = model.predict_proba(X_pred)[:, 1]
        df_resultados = df_filtrado_vaga.copy()
        df_resultados['match_score'] = (pred_probs * 100)
        df_resultados = df_resultados.sort_values(by='match_score', ascending=False)
        
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

    with tab_otimiza_vaga:
        st.header("🤖 Assistente de Otimização de Vaga")
        st.markdown("A IA analisou os CVs dos candidatos com maior *match score* e sugere melhorias para a descrição da vaga.")
        
        common_skills, recommendations = gerar_recomendacoes_vaga(df_resultados)

        st.markdown("---")
        st.subheader("💡 Palavras-chave Recomendadas")
        if recommendations:
            st.success("Para atrair candidatos mais qualificados, considere adicionar as seguintes habilidades à descrição da vaga:")
            cols = st.columns(3)
            for i, rec in enumerate(recommendations):
                with cols[i % 3]:
                    st.markdown(f"- **{rec}**")
        else:
            st.info("A descrição desta vaga já parece bem alinhada com as habilidades dos melhores candidatos. Bom trabalho!")

        with st.expander("Ver lógica da IA"):
            st.markdown("#### Como as recomendações foram geradas?")
            st.write(f"1. A IA selecionou os **{max(1, int(len(df_resultados) * 0.20))} melhores candidatos** (top 20%) com base no *match score*.")
            st.write("2. Analisou os CVs deste grupo e identificou as habilidades mais frequentes.")
            st.code(f"Habilidades mais comuns no top 20%: {', '.join(common_skills[:5])}...")
            st.write("3. Comparou essas habilidades com o texto da descrição da vaga atual.")
            st.write("4. As sugestões acima são as habilidades frequentes nos melhores candidatos que **não** foram encontradas na sua descrição.")

    with tab_otimiza_cv:
        st.header(f"📄 Assistente de Otimização de CV para a vaga: **{vaga_selecionada}**")
        st.markdown("Cole o texto do seu CV abaixo e a IA irá analisá-lo e sugerir melhorias.")
        
        cv_usuario = st.text_area("Cole o texto completo do seu CV aqui:", height=300, placeholder="Ex: Formação Acadêmica...")
        
        if st.button("Analisar meu CV", type="primary"):
            if cv_usuario and lang_model:
                with st.spinner("Analisando seu CV..."):
                    # 1. Obter texto da vaga
                    texto_vaga_completo = (str(df_filtrado_vaga['principais_atividades'].iloc[0]) + " " + str(df_filtrado_vaga['competencia_tecnicas_e_comportamentais'].iloc[0]))
                    
                    # 2. Calcular a similitude semântica
                    from sentence_transformers import util
                    embedding_vaga = lang_model.encode(texto_vaga_completo, convert_to_tensor=True)
                    embedding_cv = lang_model.encode(cv_usuario, convert_to_tensor=True)
                    score_semantico = util.cos_sim(embedding_vaga, embedding_cv).item() * 100

                    # 3. Identificar habilidades chave da vaga que faltam no CV
                    skills_list = [col.replace('skill_', '').replace('_', ' ') for col in model_columns if col.startswith('skill_')]
                    habilidades_faltantes = []
                    for skill in skills_list:
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
                    for i, skill in enumerate(habilidades_faltantes[:12]):
                        with cols_skills[i % 4]:
                            st.markdown(f"- **{skill}**")
                else:
                    st.success("Ótimo trabalho! Seu CV parece conter todas as habilidades chave mencionadas na descrição da vaga.")
            else:
                st.error("Por favor, cole o texto do seu CV para análise.")