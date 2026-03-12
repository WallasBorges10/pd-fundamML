import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Classificação de Pontes - NBI NY 2025", layout="wide")
st.title("🏗️ Classificação da Condição de Pontes (NBI NY 2025)")
st.markdown("---")

# ============================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS (CACHE)
# ============================================================
@st.cache_data
def load_and_preprocess():
    url = "https://www.fhwa.dot.gov/bridge/nbi/2025/delimited/NY25.txt"
    df = pd.read_csv(url, sep=",", quotechar="'", low_memory=False)
    df_oae = df.copy()

    # Remover colunas com mais de 50% de valores ausentes
    missing_percentage = df_oae.isnull().sum() / len(df_oae)
    cols_to_drop = missing_percentage[missing_percentage > 0.5].index
    df_oae.drop(columns=cols_to_drop, inplace=True)

    # Remover duplicatas e linhas com alvo nulo
    df_oae.drop_duplicates(inplace=True)
    df_oae.dropna(subset=['BRIDGE_CONDITION'], inplace=True)

    # Remover features de data leakage
    leaky_features = [
        'DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060',
        'CHANNEL_COND_061', 'CULVERT_COND_062', 'STRUCTURAL_EVAL_067',
        'DECK_GEOMETRY_EVAL_068', 'UNDCLRENCE_EVAL_069', 'POSTING_EVAL_070',
        'WATERWAY_EVAL_071', 'APPR_ROAD_EVAL_072', 'OPERATING_RATING_064',
        'INVENTORY_RATING_066', 'LOWEST_RATING'
    ]
    leaky_present = [col for col in leaky_features if col in df_oae.columns]
    df_sem_leakage = df_oae.drop(columns=leaky_present)

    # Selecionar features de domínio
    features_nao_condicionais = [
        'YEAR_BUILT_027', 'YEAR_RECONSTRUCTED_106', 'STRUCTURE_KIND_043A',
        'STRUCTURE_TYPE_043B', 'APPR_KIND_044A', 'APPR_TYPE_044B',
        'MAIN_UNIT_SPANS_045', 'APPR_SPANS_046', 'MAX_SPAN_LEN_MT_048',
        'STRUCTURE_LEN_MT_049', 'DECK_STRUCTURE_TYPE_107', 'SURFACE_TYPE_108A',
        'MEMBRANE_TYPE_108B', 'DECK_PROTECTION_108C', 'DESIGN_LOAD_031',
        'DEGREES_SKEW_034', 'STRUCTURE_FLARED_035', 'BRIDGE_LEN_IND_112',
        'DECK_AREA', 'PARALLEL_STRUCTURE_101', 'TEMP_STRUCTURE_103',
        'APPR_WIDTH_MT_032', 'MEDIAN_CODE_033', 'HORR_CLR_MT_047',
        'LEFT_CURB_MT_050A', 'RIGHT_CURB_MT_050B', 'ROADWAY_WIDTH_MT_051',
        'DECK_WIDTH_MT_052', 'MIN_VERT_CLR_010', 'VERT_CLR_OVER_MT_053',
        'VERT_CLR_UND_054B', 'LAT_UND_MT_055B', 'LEFT_LAT_UND_MT_056',
        'ADT_029', 'YEAR_ADT_030', 'PERCENT_ADT_TRUCK_109', 'FUTURE_ADT_114',
        'YEAR_OF_FUTURE_ADT_115', 'TRAFFIC_LANES_ON_028A', 'TRAFFIC_LANES_UND_028B',
        'ROUTE_PREFIX_005B', 'SERVICE_LEVEL_005C', 'ROUTE_NUMBER_005D',
        'DIRECTION_005E', 'FUNCTIONAL_CLASS_026', 'BASE_HWY_NETWORK_012',
        'HIGHWAY_SYSTEM_104', 'NATIONAL_NETWORK_110', 'STRAHNET_HIGHWAY_100',
        'TRAFFIC_DIRECTION_102', 'SERVICE_ON_042A', 'SERVICE_UND_042B',
        'DETOUR_KILOS_019', 'OWNER_022', 'MAINTENANCE_021', 'TOLL_020',
        'HISTORY_037', 'FEDERAL_LANDS_105', 'RAILINGS_036A', 'TRANSITIONS_036B',
        'APPR_RAIL_036C', 'APPR_RAIL_END_036D', 'BRIDGE_CONDITION'
    ]
    features_existentes = [col for col in features_nao_condicionais if col in df_sem_leakage.columns]
    df_reduzido = df_sem_leakage[features_existentes].copy()

    # Engenharia de features
    current_year = datetime.now().year
    df_reduzido['AGE'] = current_year - df_reduzido['YEAR_BUILT_027']
    df_reduzido.loc[df_reduzido['AGE'] < 0, 'AGE'] = 0

    if 'ADT_029' in df_reduzido.columns and 'TRAFFIC_LANES_ON_028A' in df_reduzido.columns:
        df_reduzido['TRAFFIC_DENSITY'] = df_reduzido['ADT_029'] / (df_reduzido['TRAFFIC_LANES_ON_028A'] + 1)
        df_reduzido['TRAFFIC_DENSITY'] = df_reduzido['TRAFFIC_DENSITY'].replace([np.inf, -np.inf], np.nan)

    df_reduzido['AGE_NORMALIZED'] = (df_reduzido['AGE'] - df_reduzido['AGE'].mean()) / df_reduzido['AGE'].std()

    # Criar target binário
    map_bridge_condition = {'G': 0, 'F': 1, 'P': 1}
    df_reduzido['TARGET'] = df_reduzido['BRIDGE_CONDITION'].map(map_bridge_condition)
    df_reduzido.dropna(subset=['TARGET'], inplace=True)
    df_reduzido['TARGET'] = df_reduzido['TARGET'].astype(int)
    df_reduzido.drop(columns=['BRIDGE_CONDITION'], inplace=True)

    return df_reduzido

with st.spinner("Carregando e processando dados... Isso pode levar alguns minutos."):
    df = load_and_preprocess()

# ============================================================
# DEFINIR COLUNAS NUMÉRICAS E CATEGÓRICAS GLOBALMENTE
# ============================================================
X_full = df.drop(columns=['TARGET'])
num_cols = X_full.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_full.select_dtypes(include=['object', 'category']).columns

# ============================================================
# SIDEBAR - NAVEGAÇÃO
# ============================================================
st.sidebar.title("Navegação")
opcao = st.sidebar.radio(
    "Ir para:",
    ["Visão Geral dos Dados", "Modelagem e Resultados", "Relatório Técnico", "Testar o Modelo"]
)

# ============================================================
# VISÃO GERAL DOS DADOS (INTERATIVA COM PLOTLY)
# ============================================================
if opcao == "Visão Geral dos Dados":
    st.header("📊 Visão Geral dos Dados (Interativa)")

    st.subheader("Primeiras linhas do dataset processado")
    st.dataframe(df.head())

    st.subheader("Distribuição da variável alvo")
    target_counts = df['TARGET'].value_counts().reset_index()
    target_counts.columns = ['Classe', 'Quantidade']
    target_counts['Classe'] = target_counts['Classe'].map({0: 'Boa (0)', 1: 'Crítica (1)'})

    fig_target = px.bar(
        target_counts,
        x='Classe',
        y='Quantidade',
        text='Quantidade',
        color='Classe',
        color_discrete_map={'Boa (0)': 'green', 'Crítica (1)': 'red'},
        title='Distribuição das Classes (Boa vs Crítica)'
    )
    fig_target.update_traces(textposition='outside')
    fig_target.update_layout(
        xaxis_title='Condição da Ponte',
        yaxis_title='Número de Observações',
        showlegend=False
    )
    st.plotly_chart(fig_target, use_container_width=True)

    st.subheader("Visão Geral de Valores Ausentes")
    st.markdown("""
    O heatmap abaixo mostra a presença de valores ausentes nas primeiras 30 colunas do dataset.
    Passe o mouse sobre os blocos para ver detalhes.
    """)
    sample_df = df.iloc[:500, :30]
    missing_matrix = sample_df.isna().astype(int)

    fig_missing = go.Figure(data=go.Heatmap(
        z=missing_matrix.T.values,
        x=missing_matrix.index,
        y=missing_matrix.columns,
        colorscale='Viridis',
        showscale=False,
        hovertemplate='Linha: %{x}<br>Coluna: %{y}<br>Ausente: %{z}<extra></extra>'
    ))
    fig_missing.update_layout(
        title='Valores Ausentes (amostra de 500 linhas x 30 colunas)',
        xaxis_title='Índice da Linha',
        yaxis_title='Colunas',
        height=600
    )
    st.plotly_chart(fig_missing, use_container_width=True)

    st.subheader("Porcentagem de Valores Ausentes por Coluna")
    missing_percent = (df.isna().sum() / len(df) * 100).reset_index()
    missing_percent.columns = ['Coluna', 'Porcentagem Ausente (%)']
    missing_percent = missing_percent.sort_values('Porcentagem Ausente (%)', ascending=False)

    fig_missing_bar = px.bar(
        missing_percent.head(20),
        x='Porcentagem Ausente (%)',
        y='Coluna',
        orientation='h',
        title='Top 20 Colunas com Maior Percentual de Valores Ausentes',
        labels={'Porcentagem Ausente (%)': '% Ausente'},
        color='Porcentagem Ausente (%)',
        color_continuous_scale='reds'
    )
    st.plotly_chart(fig_missing_bar, use_container_width=True)

    st.subheader("Estatísticas descritivas (colunas numéricas)")
    st.dataframe(df.describe())

    st.subheader("Explorar Relações entre Variáveis")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'TARGET' in numeric_cols:
        numeric_cols.remove('TARGET')

    col_x = st.selectbox("Selecione a variável do eixo X", numeric_cols, index=0)
    col_y = st.selectbox("Selecione a variável do eixo Y", numeric_cols, index=min(1, len(numeric_cols)-1))
    color_target = st.checkbox("Colorir pela classe alvo", value=True)

    # Amostra segura: usar um subconjunto com as mesmas linhas para x, y e cor
    sample_df_scatter = df.sample(min(1000, len(df)), random_state=42).copy()
    if color_target:
        sample_df_scatter['cor'] = sample_df_scatter['TARGET'].map({0: 'Boa', 1: 'Crítica'})
        fig_scatter = px.scatter(
            sample_df_scatter,
            x=col_x,
            y=col_y,
            color='cor',
            color_discrete_map={'Boa': 'green', 'Crítica': 'red'},
            opacity=0.6,
            title=f'{col_y} vs {col_x}'
        )
    else:
        fig_scatter = px.scatter(
            sample_df_scatter,
            x=col_x,
            y=col_y,
            opacity=0.6,
            title=f'{col_y} vs {col_x}'
        )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# MODELAGEM E RESULTADOS
# ============================================================
elif opcao == "Modelagem e Resultados":
    st.header("🧠 Modelagem e Resultados")

    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    num_cols_local = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols_local = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols_local),
            ('cat', categorical_transformer, cat_cols_local)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    st.subheader("Modelo Baseline: Perceptron")
    pipeline_perceptron = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', Perceptron(max_iter=1000, tol=1e-3, random_state=42, class_weight='balanced'))
    ])
    pipeline_perceptron.fit(X_train, y_train)
    y_pred_perc = pipeline_perceptron.predict(X_test)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{accuracy_score(y_test, y_pred_perc):.4f}")
    col2.metric("Precisão", f"{precision_score(y_test, y_pred_perc, zero_division=0):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred_perc):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred_perc):.4f}")

    st.subheader("Árvore de Decisão (Otimizada)")
    pipeline_dt = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5, min_samples_split=20))
    ])
    pipeline_dt.fit(X_train, y_train)
    y_pred_dt = pipeline_dt.predict(X_test)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{accuracy_score(y_test, y_pred_dt):.4f}")
    col2.metric("Precisão", f"{precision_score(y_test, y_pred_dt):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred_dt):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred_dt):.4f}")

    st.subheader("Random Forest (Otimizada)")
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42))
    ])
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{accuracy_score(y_test, y_pred_rf):.4f}")
    col2.metric("Precisão", f"{precision_score(y_test, y_pred_rf):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred_rf):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred_rf):.4f}")

    st.markdown("---")
    st.subheader("Comparação de Modelos")
    # Tenta carregar imagens salvas do notebook
    if os.path.exists("comparacao_modelos.png"):
        st.image("comparacao_modelos.png", caption="Comparação de métricas entre os modelos")
    else:
        st.warning("Imagem 'comparacao_modelos.png' não encontrada.")

    if os.path.exists("matriz_confusao.png"):
        st.image("matriz_confusao.png", caption="Matriz de confusão normalizada - Random Forest")
    else:
        st.warning("Imagem 'matriz_confusao.png' não encontrada.")

    if os.path.exists("evolucao_acuracia.png"):
        st.image("evolucao_acuracia.png", caption="Evolução da acurácia por estágio de modelagem")
    else:
        st.warning("Imagem 'evolucao_acuracia.png' não encontrada.")

    if os.path.exists("importancia_features.png"):
        st.image("importancia_features.png", caption="Importância das variáveis (Random Forest)")
    else:
        st.warning("Imagem 'importancia_features.png' não encontrada.")

# ============================================================
# RELATÓRIO TÉCNICO (com visualização e download em PDF)
# ============================================================
elif opcao == "Relatório Técnico":
    st.header("📄 Relatório Técnico")

    # Tentar download em PDF, se existir
    pdf_path = "relatorio-pd-fundamentos.pdf"
    docx_path = "relatorio-pd-fundamentos (1).docx"

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="📥 Baixar relatório completo (PDF)",
                data=file,
                file_name="relatorio_tecnico.pdf",
                mime="application/pdf"
            )
    elif os.path.exists(docx_path):
        with open(docx_path, "rb") as file:
            st.download_button(
                label="📥 Baixar relatório completo (DOCX)",
                data=file,
                file_name="relatorio_tecnico.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        st.warning("Arquivo PDF não encontrado. Disponibilizando versão DOCX.")
    else:
        st.error("Arquivo do relatório não encontrado.")

    st.markdown("""
    ### Descrição do Problema e dos Dados

    O gerenciamento de ativos de infraestrutura exige a transição de manutenções reativas para modelos preditivos. No contexto de Departamentos Nacionais de Infraestrutura, a capacidade de prever se uma Obra de Arte Especial (OAE) entrará em estado de alerta, no que tange a sua condição, permite a otimização de recursos orçamentários e a mitigação de riscos estruturais.

    **Motivação e Desafios**

    Regras fixas de inspeção exigem a complexidade multivariada (tráfego, idade, materiais) que leva à degradação. O desafio reside no ***Data Leakage***, onde variáveis calculadas de inspeção (como indicadores da condição de elementos da estrutura) podem *vazar* o resultado final para o modelo que prediz a condição da estrutura como um todo.

    **Detalhes do Dataset**

    - **Origem:** *Federal Highway Administration (FHWA)* - Dados de Nova York 2025.
    - **Volume:** 17.666 observações iniciais com 123 variáveis.
    - **Variável-Alvo:** BRIDGE_CONDITION, mapeada de forma binária: **0** para *Good* (G) e **1** para *Fair/Poor* (F, P), focando na identificação de estruturas que requerem atenção.
    - **Features:** Foram selecionadas 64 features não condicionais, incluindo idade da estrutura, densidade de tráfego (ADT), tipo de projeto e materiais.

    ### Metodologia e Modelagem

    Foram testadas três abordagens de complexidade crescente para identificar a melhor relação entre interpretabilidade e poder preditivo.

    1.  **Baseline (Perceptron):** Classificador linear para estabelecer uma métrica base.
    2.  **Árvore de Decisão (Otimizada):** Introdução de não-linearidade e regras de decisão explícitas.
    3.  **Random Forest (Final):** Modelo de *ensemble* para redução de variância e aumento do *Recall*.

    ### Resultados Principais

    - **Random Forest** alcançou o melhor desempenho com **Acurácia de 81,74%** e **Recall de 90,45%**.
    - A análise de importância das variáveis confirmou que **Idade da Estrutura** e **Volume de Tráfego (ADT)** são os fatores mais críticos.
    - A remoção de variáveis de *data leakage* foi fundamental para a generalização do modelo.

    Para mais detalhes, faça o download do relatório completo.
    """)

# ============================================================
# TESTAR O MODELO (DUAS ABAS: SELEÇÃO REAL E INPUT MANUAL)
# ============================================================
elif opcao == "Testar o Modelo":
    st.header("🔍 Testar o Modelo Random Forest")

    # Upload do modelo salvo
    uploaded_file = st.file_uploader("Carregue o arquivo do modelo treinado (modelo_pontes_ny_rf_deploy.joblib)", type=["joblib"])

    modelo = None

    if uploaded_file is not None:
        try:
            modelo = joblib.load(uploaded_file)
            st.success("Modelo carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
    else:
        st.info("Nenhum modelo carregado. O modelo será treinado automaticamente com os dados disponíveis.")

    # Se não carregou, treinar
    if modelo is None:
        with st.spinner("Treinando modelo Random Forest (pode levar alguns minutos)..."):
            X_local = df.drop(columns=['TARGET'])
            y_local = df['TARGET']
            num_cols_local = X_local.select_dtypes(include=['int64', 'float64']).columns
            cat_cols_local = X_local.select_dtypes(include=['object', 'category']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_cols_local),
                    ('cat', categorical_transformer, cat_cols_local)
                ])

            modelo = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42))
            ])
            modelo.fit(X_local, y_local)
            st.success("Modelo treinado com sucesso!")

    # Preparar dados de teste
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Criar abas
    tab1, tab2 = st.tabs(["📌 Selecionar observação real", "✏️ Inserir dados manualmente"])

    # ---------- ABA 1: Selecionar observação real ----------
    with tab1:
        st.subheader("Selecione uma observação do conjunto de teste")
        indices = X_test.index.tolist()
        selected_index = st.selectbox("Escolha o índice da observação:", indices, key="select_index")

        amostra = X_test.loc[[selected_index]]
        valor_real = y_test.loc[selected_index]

        pred = modelo.predict(amostra)[0]
        proba = modelo.predict_proba(amostra)[0]

        st.subheader("Dados da amostra selecionada")
        st.dataframe(amostra.T.rename(columns={selected_index: 'Valor'}))

        st.subheader("Resultado da Predição")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classe Real", "Crítica (1)" if valor_real == 1 else "Boa (0)")
        with col2:
            st.metric("Classe Prevista", "Crítica (1)" if pred == 1 else "Boa (0)")

        st.write(f"**Probabilidades:** Boa: {proba[0]:.2%}, Crítica: {proba[1]:.2%}")

        if st.checkbox("Mostrar predições para uma amostra aleatória de 10 linhas do teste"):
            amostra_aleatoria = X_test.sample(10, random_state=42)
            preds = modelo.predict(amostra_aleatoria)
            probs = modelo.predict_proba(amostra_aleatoria)
            resultados = pd.DataFrame({
                'Índice': amostra_aleatoria.index,
                'Classe Real': y_test.loc[amostra_aleatoria.index].values,
                'Classe Prevista': preds,
                'Prob. Boa': probs[:, 0],
                'Prob. Crítica': probs[:, 1]
            })
            st.dataframe(resultados)

    # ---------- ABA 2: Input manual dos atributos mais importantes ----------
    with tab2:
        st.subheader("Insira os valores das variáveis mais importantes")
        st.markdown("""
        As variáveis listadas abaixo são as que mais influenciam a condição da ponte, conforme análise de importância do modelo.
        Os demais atributos serão preenchidos automaticamente com valores médios (para numéricas) ou moda (para categóricas) do dataset.
        """)

        # Lista das 8 variáveis mais importantes (baseada na análise do notebook)
        top_features = [
            'AGE',                     # Idade da estrutura
            'ADT_029',                 # Tráfego médio diário
            'DECK_WIDTH_MT_052',       # Largura do deck
            'STRUCTURE_KIND_043A',     # Material da superestrutura
            'MAIN_UNIT_SPANS_045',     # Número de vãos principais
            'FUNCTIONAL_CLASS_026',    # Classe funcional da rodovia
            'MAX_SPAN_LEN_MT_048',     # Comprimento máximo do vão
            'DESIGN_LOAD_031'          # Tipo de projeto (carga)
        ]

        # Coletar inputs do usuário
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Idade da estrutura (anos)", min_value=0, max_value=200, value=50, step=1)
            adt = st.number_input("Tráfego médio diário (ADT)", min_value=0, value=5000, step=100)
            deck_width = st.number_input("Largura do deck (m)", min_value=0.0, value=10.0, step=0.1)
            structure_kind = st.selectbox("Material da superestrutura", options=[1,2,3,4,5,6,7,8,9], index=0)
        with col2:
            main_spans = st.number_input("Número de vãos principais", min_value=1, value=2, step=1)
            func_class = st.selectbox("Classe funcional", options=[1,2,3,4,5,6,7,8,9,11,12,14,16,17,19], index=0)
            max_span_len = st.number_input("Comprimento máximo do vão (m)", min_value=0.0, value=20.0, step=0.1)
            design_load = st.selectbox("Carga de projeto", options=[1,2,3,4,5,6,7,8,9], index=0)

        # Criar um DataFrame com uma única linha, preenchendo todas as features
        # Primeiro, criamos um dicionário com os valores médios/moda para todas as colunas
        default_row = {}
        for col in X.columns:
            if col in top_features:
                # Será substituído depois
                default_row[col] = None
            else:
                if col in num_cols:  # num_cols global
                    default_row[col] = X[col].median()
                else:
                    default_row[col] = X[col].mode()[0] if not X[col].mode().empty else None

        # Agora inserimos os valores fornecidos pelo usuário
        default_row['AGE'] = age
        default_row['ADT_029'] = adt
        default_row['DECK_WIDTH_MT_052'] = deck_width
        default_row['STRUCTURE_KIND_043A'] = structure_kind
        default_row['MAIN_UNIT_SPANS_045'] = main_spans
        default_row['FUNCTIONAL_CLASS_026'] = func_class
        default_row['MAX_SPAN_LEN_MT_048'] = max_span_len
        default_row['DESIGN_LOAD_031'] = design_load

        # Calcular AGE_NORMALIZED com base na média e desvio do dataset
        default_row['AGE_NORMALIZED'] = (age - df['AGE'].mean()) / df['AGE'].std()

        # Se ADT_029 e TRAFFIC_LANES_ON_028A existirem, calcular TRAFFIC_DENSITY
        if 'ADT_029' in X.columns and 'TRAFFIC_LANES_ON_028A' in X.columns:
            lanes = default_row.get('TRAFFIC_LANES_ON_028A', 1)
            default_row['TRAFFIC_DENSITY'] = adt / (lanes + 1)

        # Converter para DataFrame
        input_df = pd.DataFrame([default_row])

        st.subheader("Dados para predição")
        st.dataframe(input_df.T.rename(columns={0: 'Valor'}))

        if st.button("Classificar"):
            pred = modelo.predict(input_df)[0]
            proba = modelo.predict_proba(input_df)[0]

            st.subheader("Resultado da Predição")
            if pred == 0:
                st.success("**Classe prevista: Boa (0)**")
            else:
                st.error("**Classe prevista: Crítica (1)**")
            st.write(f"**Probabilidades:** Boa: {proba[0]:.2%}, Crítica: {proba[1]:.2%}")
