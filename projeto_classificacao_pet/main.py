
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from pre_processamento import preparar_dataset_pre_organizado, criar_geradores_de_dados
from modelos import criar_cnn_personalizada, criar_modelo_transfer_learning
from treinamento_e_avaliacao import treinar_modelo, gerar_graficos_historico, avaliar_modelo

def main():
    print("Iniciando o processo de classificação de cães e gatos...")

    caminho_base_dados = "projeto_classificacao_pet/dados"
    caminho_treino_dados = os.path.join(caminho_base_dados, "treino")
    caminho_validacao_dados = os.path.join(caminho_base_dados, "validacao")
    caminho_arquivo_dataset = r"C:\Engenharia de Software\Redes Neurais\archive\catsvsdogs\train"

    # 1. Preparar o dataset
    if not (os.path.exists(os.path.join(caminho_treino_dados, "cachorros")) and
            os.path.exists(os.path.join(caminho_treino_dados, "gatos")) and
            os.path.exists(os.path.join(caminho_validacao_dados, "cachorros")) and
            os.path.exists(os.path.join(caminho_validacao_dados, "gatos")) and
            len(os.listdir(os.path.join(caminho_treino_dados, "cachorros"))) > 0):
        print("Dataset não preparado. Executando preparação...")
        if not preparar_dataset_pre_organizado(caminho_base_dados, caminho_origem=caminho_arquivo_dataset):
            print("Erro na preparação do dataset. Abortando.")
            return
    else:
        print("Pastas de treino e validação já estão prontas.")

    # 2. Criar geradores de dados
    tamanho_imagem = (150, 150)
    tamanho_batch = 32
    gerador_treino, gerador_validacao = criar_geradores_de_dados(
        caminho_treino_dados, caminho_validacao_dados, tamanho_imagem, tamanho_batch
    )

    # Callbacks para melhor acurácia
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

    # 3. Modelo CNN Personalizada
    print("\n--- Processando CNN Personalizada ---")
    modelo_cnn = criar_cnn_personalizada(input_shape=(tamanho_imagem[0], tamanho_imagem[1], 3))
    modelo_cnn._name = "cnn_personalizada" # Definir nome para o modelo
    historico_cnn = treinar_modelo(modelo_cnn, gerador_treino, gerador_validacao, epochs=10, callbacks=[early_stopping, reduce_lr])
    gerar_graficos_historico(historico_cnn, "CNN_Personalizada")
    avaliar_modelo(modelo_cnn, gerador_validacao, "CNN_Personalizada")
    modelo_cnn.save(os.path.join("projeto_classificacao_pet/modelos", "cnn_personalizada.keras"))

    # 4. Modelo Transfer Learning (VGG16)
    print("\n--- Processando Modelo de Transfer Learning (VGG16) ---")
    modelo_tl = criar_modelo_transfer_learning(input_shape=(tamanho_imagem[0], tamanho_imagem[1], 3))
    modelo_tl._name = "transfer_learning_vgg16" # Definir nome para o modelo
    historico_tl = treinar_modelo(modelo_tl, gerador_treino, gerador_validacao, epochs=10, callbacks=[early_stopping, reduce_lr])
    gerar_graficos_historico(historico_tl, "Transfer_Learning_VGG16")
    avaliar_modelo(modelo_tl, gerador_validacao, "Transfer_Learning_VGG16")
    modelo_tl.save(os.path.join("projeto_classificacao_pet/modelos", "transfer_learning_vgg16.keras"))

    print("\nProcesso de classificação concluído. Verifique a pasta \"projeto_classificacao_pet/saidas\" para os resultados.")

if __name__ == "__main__":
    main()
