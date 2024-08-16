import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objs as go
from PIL import Image  # Import Image module from PIL (Pillow)

# Load the DataFrame and Model
df_validacao = pd.read_csv('df_validacao.csv')
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer used during training
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Custom function to add symbols in "DIFERENÇA" column
def add_custom_symbols(df):
    df['DIFERENCA'] = df['DIFERENCA'].apply(lambda x: '✓' if x else 'X')
    return df

# Apply custom symbols to the "DIFERENÇA" column
df_validacao = add_custom_symbols(df_validacao)

# App Layout
st.title("Análise de Ponto de Virada")

# Top Section with Name, Course, and Registration
st.sidebar.header("Informações do Estudante")
st.sidebar.text("Nome: Gabriel Hagle Freesz")
st.sidebar.text("Curso: [Nome do Curso]")
st.sidebar.text("Matrícula: rm351357")


# Title of the Project
st.header("Projeto de Predição: Ponto de Virada")

# Navigation Buttons
page = st.sidebar.selectbox("Escolha a Análise", ["Análise Exploratória", "Modelo de Predição"])

if page == "Análise Exploratória":
    st.subheader("Análise Exploratória")
    # Insert your plotting code here

    # Load and display images from a directory
    image_dir = "plots/"  # Update this path to your directory
    image_files = ["idade.png", "instituicao_ensino.png", "output.png","pedra.png","tempo_na)ong.png","turmas.png"]  # List of your image files
    images = [
        {
            "filename": "idade.png",
            "title": "Gráfico de Distribuição de Idade",
            "caption": "Idade x Ano",
            "description": "Este gráfico mostra a distribuição das idades dos alunos ao longo dos anos 2020, 2021 e 2022. A maioria dos alunos tem entre 9 e 14 anos, com picos notáveis em 10 anos de idade, indicando que essa faixa etária é a mais representativa entre os alunos da instituição. A distribuição é semelhante nos três anos, com uma leve queda em 2021."
        },
        {
            "filename": "instituicao_ensino.png",
            "title": "Gráfico de Distribuição de Instituição de Ensino",
            "caption": "Instituição de Ensino x Ano",
            "description": "Este gráfico compara as instituições de ensino dos alunos nos anos 2020, 2021 e 2022. A maioria dos alunos é de escolas públicas, especialmente nos anos de 2020 e 2021. Outras instituições têm representações menores, indicando uma concentração significativa em escolas públicas."
        },
        {
            "filename": "output.png",
            "title": "Gráfico de Distribuição de Fase",
            "caption": "Fase x Ano",
            "description": "Este gráfico ilustra a distribuição das fases dos alunos ao longo dos anos 2020, 2021 e 2022. A fase 0.0 tem a maior quantidade de alunos, seguida das fases 1.0, 2.0 e 3.0, com uma diminuição progressiva nas fases superiores. O padrão é relativamente consistente entre os anos."
        },
        {
            "filename": "pedra.png",
            "title": "Gráfico de Distribuição de Pedra",
            "caption": "Pedra x Ano",
            "description": "Este gráfico representa a distribuição dos alunos por grupos (Pedra) nos anos de 2020, 2021 e 2022. Os grupos 'Ametista' e 'Ágata' têm o maior número de alunos, enquanto os grupos 'Topázio' e 'Quartzo' apresentam números menores. A distribuição varia um pouco entre os anos, com uma leve diminuição em alguns grupos em 2021."
        },
        {
            "filename": "tempo_na)ong.png",
            "title": "Gráfico de Distribuição da Quantidade de Anos na ONG",
            "caption": "Anos na ONG x Ano",
            "description": "Este gráfico mostra a distribuição dos alunos de acordo com os anos de participação no programa Passos Mágicos em 2020, 2021 e 2022. A maioria dos alunos tem 1 ou 2 anos no programa, com um número menor de alunos com mais de 3 anos. Em 2020, há uma distribuição clara, enquanto nos outros anos a participação é significativamente menor."
        },
        {
            "filename": "turmas.png",
            "title": "Gráfico de Distribuição de Turma",
            "caption": "Turma x Ano",
            "description": "Este gráfico apresenta a distribuição das turmas dos alunos ao longo dos anos 2020, 2021 e 2022. As turmas A, B, e C têm o maior número de alunos, especialmente nos anos de 2020 e 2021. Algumas turmas como M e N mostram um aumento significativo em 2022, indicando uma possível reorganização ou expansão das turmas ao longo do tempo."
        }
    ]
    
    for img in images:
        image_path = image_dir + img["filename"]
        image = Image.open(image_path)
        
        st.image(image, caption=img["caption"], use_column_width=True)
        st.markdown(f"### {img['title']}")
        st.markdown(img["description"])
        st.markdown("---")  # Separator line

elif page == "Modelo de Predição":
    st.subheader("Modelo de Predição")

    # Input Fields for Prediction
    destaque_ieg = st.text_area("Destaque IEG")
    destaque_ida = st.text_area("Destaque IDA")
    destaque_ipv = st.text_area("Destaque IPV")
    inde = st.number_input("INDE", min_value=0.0, max_value=10.0, step=0.01)

    # Analyze Button
    if st.button("Analisar"):
        # Process the inputs and make prediction
        input_data = {
            'DESTAQUE_IEG': [destaque_ieg], 
            'DESTAQUE_IDA': [destaque_ida], 
            'DESTAQUE_IPV': [destaque_ipv], 
            'INDE': [inde]
        }
        input_df = pd.DataFrame(input_data)

        # Vectorize the text fields
        input_text_combined = input_df['DESTAQUE_IEG'] + " " + input_df['DESTAQUE_IDA'] + " " + input_df['DESTAQUE_IPV']
        input_text_vectorized = vectorizer.transform(input_text_combined)

        # Combine vectorized text data with numeric data
        input_final = pd.concat([pd.DataFrame(input_text_vectorized.toarray()), input_df[['INDE']].reset_index(drop=True)], axis=1)

        # Make prediction using the model
        prediction = model.predict(input_final)

        if prediction[0] == 1:
            st.success("Atingiu o Ponto de Virada")
        else:
            st.error("Não atingiu o Ponto de Virada")

        # Plot the effect of INDE on prediction using Plotly
        inde_values = np.linspace(0, 10, 100)
        predictions = []

        for inde_value in inde_values:
            input_df['INDE'] = inde_value
            input_final = pd.concat([pd.DataFrame(input_text_vectorized.toarray()), input_df[['INDE']].reset_index(drop=True)], axis=1)
            pred = model.predict(input_final)
            predictions.append(pred[0])

        # Apply smoothing
        def smooth_predictions(inde_values, predictions):
            smoothed_preds = []
            flag = False
            for pred in predictions:
                if pred == 1:
                    flag = True
                if flag:
                    smoothed_preds.append(1)
                else:
                    smoothed_preds.append(pred)
            return smoothed_preds

        smoothed_predictions = smooth_predictions(inde_values, predictions)

        # Create a plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inde_values, y=smoothed_predictions, mode='lines+markers',
                                 marker=dict(color='blue'), name='Previsão'))

        fig.update_layout(
            title='Impacto do INDE sobre o Ponto de Virada (Com Smoothing)',
            xaxis_title='Valor do INDE',
            yaxis_title='Previsão (0 = Não Atingiu, 1 = Atingiu)',
            hovermode='x unified'
        )

        # Display the plot
        st.plotly_chart(fig)

    # Display the validation DataFrame with filtering options
    st.subheader("Casos Analisados")
    
    # Add filtering options
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        filtro_nome = st.text_input("Filtrar por Nome")
    with col2:
        filtro_inde = st.text_input("Filtrar por INDE")
    with col3:
        filtro_ponto_virada = st.selectbox("Filtrar por Ponto de Virada", options=["", "Sim", "Não"])
    with col4:
        filtro_pred_ponto_virada = st.selectbox("Filtrar por Previsão", options=["", "Sim", "Não"])
    with col5:
        filtro_diferenca = st.selectbox("Filtrar por Diferença", options=["", "✓", "X"])
    
    # Apply the filters
    if filtro_nome:
        df_validacao = df_validacao[df_validacao['NOME'] == filtro_nome]
    if filtro_inde:
        df_validacao = df_validacao[df_validacao['INDE'].astype(str).str.contains(filtro_inde)]
    if filtro_ponto_virada:
        df_validacao = df_validacao[df_validacao['PONTO_VIRADA'] == filtro_ponto_virada]
    if filtro_pred_ponto_virada:
        df_validacao = df_validacao[df_validacao['PRED_PONTO_VIRADA'] == filtro_pred_ponto_virada]
    if filtro_diferenca:
        df_validacao = df_validacao[df_validacao['DIFERENCA'] == filtro_diferenca]

    # Show the DataFrame
    st.dataframe(df_validacao)

# Set Layout Style
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to bottom, #ADD8E6, #0000FF);
    }
    .sidebar .sidebar-content {
        background: #B0C4DE;
    }
    .stButton > button {
        color: #FFF;
        background: #4682B4;
    }
    </style>
    """,
    unsafe_allow_html=True
)
