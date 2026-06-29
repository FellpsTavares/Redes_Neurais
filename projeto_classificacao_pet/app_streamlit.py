
import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Classificação de Cães e Gatos",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("🐾 Sistema de Classificação de Cães e Gatos")
st.markdown("""
Este painel interativo apresenta os resultados de um sistema de classificação de cães e gatos 
utilizando duas arquiteturas de redes neurais: uma CNN personalizada e um modelo de Transfer Learning (VGG16).
""")

# Diretórios
caminho_saidas = "projeto_classificacao_pet/saidas"
caminho_modelos = "projeto_classificacao_pet/modelos"
caminho_dados = "projeto_classificacao_pet/dados"

# Carregar modelos
@st.cache_resource
def carregar_modelos():
    try:
        modelo_cnn = tf.keras.models.load_model(os.path.join(caminho_modelos, "cnn_personalizada.keras"))
        modelo_tl = tf.keras.models.load_model(os.path.join(caminho_modelos, "transfer_learning_vgg16.keras"))
        return modelo_cnn, modelo_tl
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None, None

# Sidebar para navegação
st.sidebar.title("Navegação")
opcao = st.sidebar.radio("Selecione uma opção:", 
    ["Visão Geral", "Comparação de Modelos", "Predição em Tempo Real", "Análise Detalhada"])

# ===== PÁGINA 1: VISÃO GERAL =====
if opcao == "Visão Geral":
    st.header("📊 Visão Geral do Projeto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Arquitetura CNN Personalizada")
        st.markdown("""
        - **Camadas Convolucionais**: 3 camadas com 32, 64 e 128 filtros
        - **Pooling**: MaxPooling2D após cada convolução
        - **Camadas Densas**: 512 neurônios com Dropout (0.5)
        - **Ativação**: ReLU (convoluções) e Sigmoid (saída)
        - **Otimizador**: Adam
        - **Loss**: Binary Crossentropy
        """)
    
    with col2:
        st.subheader("Transfer Learning (VGG16)")
        st.markdown("""
        - **Modelo Base**: VGG16 pré-treinado no ImageNet
        - **Camadas Congeladas**: Todas as camadas do modelo base
        - **Camadas Personalizadas**: Flatten + Dense(256) + Dropout(0.5) + Dense(1)
        - **Ativação**: ReLU (dense) e Sigmoid (saída)
        - **Otimizador**: Adam
        - **Loss**: Binary Crossentropy
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Estatísticas do Dataset")
    
    # Contar imagens no dataset
    treino_cachorros = len(os.listdir(os.path.join(caminho_dados, "treino", "cachorros"))) if os.path.exists(os.path.join(caminho_dados, "treino", "cachorros")) else 0
    treino_gatos = len(os.listdir(os.path.join(caminho_dados, "treino", "gatos"))) if os.path.exists(os.path.join(caminho_dados, "treino", "gatos")) else 0
    validacao_cachorros = len(os.listdir(os.path.join(caminho_dados, "validacao", "cachorros"))) if os.path.exists(os.path.join(caminho_dados, "validacao", "cachorros")) else 0
    validacao_gatos = len(os.listdir(os.path.join(caminho_dados, "validacao", "gatos"))) if os.path.exists(os.path.join(caminho_dados, "validacao", "gatos")) else 0
    
    dados_dataset = {
        "Conjunto": ["Treino", "Treino", "Validação", "Validação"],
        "Classe": ["Cachorros", "Gatos", "Cachorros", "Gatos"],
        "Quantidade": [treino_cachorros, treino_gatos, validacao_cachorros, validacao_gatos]
    }
    
    df_dataset = pd.DataFrame(dados_dataset)
    st.dataframe(df_dataset, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Treino - Cachorros", treino_cachorros)
    with col2:
        st.metric("Treino - Gatos", treino_gatos)
    with col3:
        st.metric("Validação - Cachorros", validacao_cachorros)
    with col4:
        st.metric("Validação - Gatos", validacao_gatos)

# ===== PÁGINA 2: COMPARAÇÃO DE MODELOS =====
elif opcao == "Comparação de Modelos":
    st.header("🔄 Comparação de Modelos")
    
    col1, col2 = st.columns(2)
    
    # CNN Personalizada
    with col1:
        st.subheader("CNN Personalizada")
        if os.path.exists(os.path.join(caminho_saidas, "historico_CNN_Personalizada.png")):
            img = Image.open(os.path.join(caminho_saidas, "historico_CNN_Personalizada.png"))
            st.image(img, use_container_width=True)
        else:
            st.warning("Gráfico de histórico não encontrado")
        
        if os.path.exists(os.path.join(caminho_saidas, "matriz_confusao_CNN_Personalizada.png")):
            img = Image.open(os.path.join(caminho_saidas, "matriz_confusao_CNN_Personalizada.png"))
            st.image(img, use_container_width=True)
        else:
            st.warning("Matriz de confusão não encontrada")
        
        if os.path.exists(os.path.join(caminho_saidas, "relatorio_classificacao_CNN_Personalizada.txt")):
            with open(os.path.join(caminho_saidas, "relatorio_classificacao_CNN_Personalizada.txt"), "r") as f:
                relatorio = f.read()
            st.text_area("Relatório de Classificação", relatorio, height=200, disabled=True)
    
    # Transfer Learning
    with col2:
        st.subheader("Transfer Learning (VGG16)")
        if os.path.exists(os.path.join(caminho_saidas, "historico_Transfer_Learning_VGG16.png")):
            img = Image.open(os.path.join(caminho_saidas, "historico_Transfer_Learning_VGG16.png"))
            st.image(img, use_container_width=True)
        else:
            st.warning("Gráfico de histórico não encontrado")
        
        if os.path.exists(os.path.join(caminho_saidas, "matriz_confusao_Transfer_Learning_VGG16.png")):
            img = Image.open(os.path.join(caminho_saidas, "matriz_confusao_Transfer_Learning_VGG16.png"))
            st.image(img, use_container_width=True)
        else:
            st.warning("Matriz de confusão não encontrada")
        
        if os.path.exists(os.path.join(caminho_saidas, "relatorio_classificacao_Transfer_Learning_VGG16.txt")):
            with open(os.path.join(caminho_saidas, "relatorio_classificacao_Transfer_Learning_VGG16.txt"), "r") as f:
                relatorio = f.read()
            st.text_area("Relatório de Classificação", relatorio, height=200, disabled=True)

# ===== PÁGINA 3: PREDIÇÃO EM TEMPO REAL =====
elif opcao == "Predição em Tempo Real":
    st.header("🔮 Predição em Tempo Real")
    
    modelo_cnn, modelo_tl = carregar_modelos()
    
    if modelo_cnn is None or modelo_tl is None:
        st.error("Não foi possível carregar os modelos.")
    else:
        uploaded_file = st.file_uploader("Carregue uma imagem de um cão ou gato", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Exibir a imagem
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem Carregada", use_container_width=True)
            
            # Pré-processar a imagem
            img_array = img_to_array(image.resize((150, 150))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Fazer predições
            pred_cnn = modelo_cnn.predict(img_array, verbose=0)[0][0]
            pred_tl = modelo_tl.predict(img_array, verbose=0)[0][0]
            
            # Exibir resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("CNN Personalizada")
                if pred_cnn > 0.5:
                    st.success(f"🐕 Cachorro ({pred_cnn*100:.2f}%)")
                else:
                    st.success(f"🐈 Gato ({(1-pred_cnn)*100:.2f}%)")
                st.progress(pred_cnn)
            
            with col2:
                st.subheader("Transfer Learning (VGG16)")
                if pred_tl > 0.5:
                    st.success(f"🐕 Cachorro ({pred_tl*100:.2f}%)")
                else:
                    st.success(f"🐈 Gato ({(1-pred_tl)*100:.2f}%)")
                st.progress(pred_tl)

# ===== PÁGINA 4: ANÁLISE DETALHADA =====
elif opcao == "Análise Detalhada":
    st.header("📋 Análise Detalhada")
    
    st.subheader("Métricas Comparativas")
    
    # Ler relatórios
    try:
        with open(os.path.join(caminho_saidas, "relatorio_classificacao_CNN_Personalizada.txt"), "r") as f:
            relatorio_cnn = f.read()
        
        with open(os.path.join(caminho_saidas, "relatorio_classificacao_Transfer_Learning_VGG16.txt"), "r") as f:
            relatorio_tl = f.read()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CNN Personalizada")
            st.text(relatorio_cnn)
        
        with col2:
            st.subheader("Transfer Learning (VGG16)")
            st.text(relatorio_tl)
    
    except FileNotFoundError:
        st.error("Relatórios de classificação não encontrados.")
    
    st.markdown("---")
    
    st.subheader("Conclusões")
    st.markdown("""
    ### Observações Importantes:
    
    1. **Dataset Pequeno**: O dataset de demonstração contém apenas algumas imagens, o que resulta em métricas baixas.
    2. **Treinamento Limitado**: Apenas 2 epochs foram utilizados para demonstração rápida.
    3. **Recomendações para Melhoria**:
       - Usar o dataset completo do Kaggle (>20.000 imagens)
       - Aumentar o número de epochs (10-50)
       - Implementar técnicas avançadas de data augmentation
       - Ajustar hiperparâmetros (learning_rate, batch_size)
       - Considerar fine-tuning do modelo VGG16
    
    ### Próximos Passos:
    - Treinar com o dataset completo
    - Implementar validação cruzada
    - Explorar outras arquiteturas (ResNet, EfficientNet)
    - Implementar ensemble de modelos
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Sistema de Classificação de Cães e Gatos | Desenvolvido com TensorFlow e Streamlit</p>
</div>
""", unsafe_allow_html=True)
