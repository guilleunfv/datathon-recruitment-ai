# 🤖 Decision AI: Otimizador de Recrutamento

## 📝 Descrição do Projeto

Decision AI é uma solução de Inteligência Artificial desenvolvida para o Datathon da Pós-Tech FIAP, com o objetivo de otimizar e agilizar o processo de recrutamento e seleção da empresa Decision. A ferramenta analisa dados históricos de candidaturas para ranquear candidatos, otimizar a descrição de vagas e ajudar candidatos a melhorarem seus CVs.

### ✨ Funcionalidades Principais

1.  **Ranking de Candidatos:** Utiliza um modelo de Machine Learning (XGBoost) para calcular um "Match Score", ranqueando os candidatos mais promissores para cada vaga.
2.  **Otimização de Vaga com IA:** Analisa o perfil dos melhores candidatos e sugere palavras-chave e habilidades que podem ser adicionadas à descrição da vaga para atrair talentos mais qualificados.
3.  **Otimização de CV para Candidatos:** Permite que um candidato cole seu CV e receba um feedback instantâneo sobre seu alinhamento com a vaga e sugestões de habilidades a serem incluídas.

## 🚀 Link para a Aplicação (Demo)

**[Acesse a aplicação aqui!](URL_DA_SUA_APP_STREAMLIT)**

## 🛠️ Stack Tecnológica

- **Linguagem:** Python 3.11
- **Análise e Modelagem:** Pandas, Scikit-learn, XGBoost, Sentence-Transformers
- **Aplicação Web:** Streamlit
- **Cloud:** Google Cloud Storage (GCS) para armazenamento de dados e modelos.
- **Ambiente:** Gerenciado com `venv` e `pip`.

## ⚙️ Como Rodar o App Localmente

Siga os passos abaixo para executar a aplicação no seu ambiente local.

### Pré-requisitos

- Python 3.11
- Git
- Conta Google Cloud com um projeto e um bucket criados.

### Passos

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/datathon-recruitment-ai.git
    cd datathon-recruitment-ai
    ```

2.  **Crie e ative o ambiente virtual:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as credenciais do Google Cloud:**
    - Crie uma conta de serviço no seu projeto GCP com o papel "Visualizador de Objetos do Storage".
    - Gere uma chave JSON para esta conta.
    - Crie o arquivo `.streamlit/secrets.toml` e adicione as credenciais no formato especificado [aqui](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management#connect-streamlit-community-cloud-to-your-private-google-cloud-storage-bucket).

5.  **Execute a aplicação:**
    ```bash
    streamlit run app/app.py
    ```

## 🔄 Como Retreinar o Modelo

Para retreinar o modelo com novos dados, basta executar o script principal de treinamento. (Requer configuração de credenciais GCS no ambiente local).

```bash
python src/train.py