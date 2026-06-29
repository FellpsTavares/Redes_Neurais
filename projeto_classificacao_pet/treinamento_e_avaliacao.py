
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import tensorflow as tf

def treinar_modelo(modelo, gerador_treino, gerador_validacao, epochs=10, callbacks=None):
    print(f"Iniciando treinamento do modelo: {modelo.name}")
    historico = modelo.fit(
        gerador_treino,
        steps_per_epoch=gerador_treino.samples // gerador_treino.batch_size,
        epochs=epochs,
        validation_data=gerador_validacao,
        validation_steps=gerador_validacao.samples // gerador_validacao.batch_size,
        callbacks=callbacks
    )
    print(f"Treinamento do modelo {modelo.name} concluído.")
    return historico

def gerar_graficos_historico(historico, nome_modelo, caminho_saida="projeto_classificacao_pet/saidas"):
    print(f"Gerando gráficos de histórico para o modelo {nome_modelo}...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history["accuracy"], label="Accuracy de Treino")
    plt.plot(historico.history["val_accuracy"], label="Accuracy de Validação")
    plt.title(f"Accuracy do Modelo {nome_modelo}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historico.history["loss"], label="Loss de Treino")
    plt.plot(historico.history["val_loss"], label="Loss de Validação")
    plt.title(f"Loss do Modelo {nome_modelo}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(caminho_saida, f"historico_{nome_modelo}.png"))
    plt.close()
    print(f"Gráficos de histórico para {nome_modelo} salvos em {caminho_saida}")

def avaliar_modelo(modelo, gerador_validacao, nome_modelo, caminho_saida="projeto_classificacao_pet/saidas"):
    print(f"Avaliando o modelo {nome_modelo}...")
    
    # Prever as classes no conjunto de validação
    gerador_validacao.reset()
    predicoes = modelo.predict(gerador_validacao, steps=gerador_validacao.samples // gerador_validacao.batch_size + 1)
    rotulos_preditos = (predicoes > 0.5).astype(int)
    rotulos_verdadeiros = gerador_validacao.classes

    # Ajustar os rótulos verdadeiros para corresponder ao número de amostras após o reset
    # Isso é necessário porque o gerador pode retornar um número diferente de amostras
    # se o batch_size não dividir exatamente o número total de amostras.
    rotulos_verdadeiros = rotulos_verdadeiros[:len(rotulos_preditos)]

    # Matriz de Confusão
    matriz_conf = confusion_matrix(rotulos_verdadeiros, rotulos_preditos)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_conf, annot=True, fmt="d", cmap="Blues",
                xticklabels=gerador_validacao.class_indices.keys(),
                yticklabels=gerador_validacao.class_indices.keys())
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.savefig(os.path.join(caminho_saida, f"matriz_confusao_{nome_modelo}.png"))
    plt.close()
    print(f"Matriz de Confusão para {nome_modelo} salva em {caminho_saida}")

    # Relatório de Classificação
    relatorio_classificacao = classification_report(rotulos_verdadeiros, rotulos_preditos, target_names=gerador_validacao.class_indices.keys())
    print(f"\nRelatório de Classificação - {nome_modelo}:\n")
    print(relatorio_classificacao)

    with open(os.path.join(caminho_saida, f"relatorio_classificacao_{nome_modelo}.txt"), "w") as f:
        f.write(relatorio_classificacao)
    print(f"Relatório de Classificação para {nome_modelo} salvo em {caminho_saida}")

    return relatorio_classificacao


if __name__ == "__main__":
    # Este bloco é apenas para teste e demonstração
    # Em um cenário real, os geradores e modelos seriam passados do script principal
    from pre_processamento import criar_geradores_de_dados, preparar_dataset_pre_organizado
    from modelos import criar_cnn_personalizada, criar_modelo_transfer_learning

    caminho_base_dados = "projeto_classificacao_pet/dados"
    caminho_treino_dados = os.path.join(caminho_base_dados, "treino")
    caminho_validacao_dados = os.path.join(caminho_base_dados, "validacao")

    # Certifique-se de que o dataset está preparado
    if not (os.path.exists(os.path.join(caminho_treino_dados, "cachorros")) and 
            os.path.exists(os.path.join(caminho_treino_dados, "gatos")) and 
            os.path.exists(os.path.join(caminho_validacao_dados, "cachorros")) and 
            os.path.exists(os.path.join(caminho_validacao_dados, "gatos")) and 
            len(os.listdir(os.path.join(caminho_treino_dados, "cachorros"))) > 0):
        print("Dataset não preparado. Executando preparação...")
        preparar_dataset_pre_organizado(caminho_base_dados)

    gerador_treino, gerador_validacao = criar_geradores_de_dados(caminho_treino_dados, caminho_validacao_dados, tamanho_imagem=(150, 150), tamanho_batch=32)

    # Modelo CNN Personalizada
    print("\n--- Treinando e Avaliando CNN Personalizada ---")
    modelo_cnn = criar_cnn_personalizada()
    historico_cnn = treinar_modelo(modelo_cnn, gerador_treino, gerador_validacao, epochs=1)
    gerar_graficos_historico(historico_cnn, "CNN_Personalizada")
    avaliar_modelo(modelo_cnn, gerador_validacao, "CNN_Personalizada")

    # Modelo Transfer Learning
    print("\n--- Treinando e Avaliando Modelo de Transfer Learning (VGG16) ---")
    modelo_tl = criar_modelo_transfer_learning()
    historico_tl = treinar_modelo(modelo_tl, gerador_treino, gerador_validacao, epochs=1)
    gerar_graficos_historico(historico_tl, "Transfer_Learning_VGG16")
    avaliar_modelo(modelo_tl, gerador_validacao, "Transfer_Learning_VGG16")
