
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random

def preparar_dataset_pre_organizado(caminho_base='projeto_classificacao_pet/dados', proporcao_validacao=0.2, caminho_origem=None):
    print('Preparando o dataset pré-organizado em pastas de treino e validação...')

    if caminho_origem is not None:
        caminho_origem_gatos = os.path.join(caminho_origem, 'cats')
        caminho_origem_cachorros = os.path.join(caminho_origem, 'dogs')
    else:
        caminho_origem_gatos = os.path.join(caminho_base, 'Cats')
        caminho_origem_cachorros = os.path.join(caminho_base, 'Dogs')

    if not os.path.exists(caminho_origem_gatos) or not os.path.exists(caminho_origem_cachorros):
        print(f'ERRO: Pastas de origem não encontradas. Verifique:\n  {caminho_origem_gatos}\n  {caminho_origem_cachorros}')
        return False

    caminho_treino = os.path.join(caminho_base, 'treino')
    caminho_validacao = os.path.join(caminho_base, 'validacao')

    os.makedirs(os.path.join(caminho_treino, 'cachorros'), exist_ok=True)
    os.makedirs(os.path.join(caminho_treino, 'gatos'), exist_ok=True)
    os.makedirs(os.path.join(caminho_validacao, 'cachorros'), exist_ok=True)
    os.makedirs(os.path.join(caminho_validacao, 'gatos'), exist_ok=True)

    def copiar_arquivos(origem_pasta, destino_treino_pasta, destino_validacao_pasta):
        arquivos = [os.path.join(origem_pasta, f) for f in os.listdir(origem_pasta) if os.path.isfile(os.path.join(origem_pasta, f))]
        random.shuffle(arquivos)
        num_validacao = int(len(arquivos) * proporcao_validacao)

        for i, arquivo in enumerate(arquivos):
            if i < num_validacao:
                shutil.copy2(arquivo, destino_validacao_pasta)
            else:
                shutil.copy2(arquivo, destino_treino_pasta)

    print('Copiando imagens de gatos...')
    copiar_arquivos(caminho_origem_gatos, os.path.join(caminho_treino, 'gatos'), os.path.join(caminho_validacao, 'gatos'))
    print('Copiando imagens de cachorros...')
    copiar_arquivos(caminho_origem_cachorros, os.path.join(caminho_treino, 'cachorros'), os.path.join(caminho_validacao, 'cachorros'))

    print('Dataset organizado com sucesso.')
    return True

def criar_geradores_de_dados(caminho_treino, caminho_validacao, tamanho_imagem=(150, 150), tamanho_batch=32):
    print('Criando geradores de dados com data augmentation...')

    gerador_aumento_dados_treino = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    gerador_validacao = ImageDataGenerator(rescale=1./255)

    gerador_treino = gerador_aumento_dados_treino.flow_from_directory(
        caminho_treino,
        target_size=tamanho_imagem,
        batch_size=tamanho_batch,
        class_mode='binary'
    )

    gerador_validacao = gerador_validacao.flow_from_directory(
        caminho_validacao,
        target_size=tamanho_imagem,
        batch_size=tamanho_batch,
        class_mode='binary'
    )

    print('Geradores de dados criados.')
    return gerador_treino, gerador_validacao

if __name__ == '__main__':
    caminho_base_dados = 'projeto_classificacao_pet/dados'
    caminho_treino_dados = os.path.join(caminho_base_dados, 'treino')
    caminho_validacao_dados = os.path.join(caminho_base_dados, 'validacao')

    # Verifica se as pastas de treino/validacao já estão preenchidas
    if not (os.path.exists(os.path.join(caminho_treino_dados, 'cachorros')) and 
            os.path.exists(os.path.join(caminho_treino_dados, 'gatos')) and 
            os.path.exists(os.path.join(caminho_validacao_dados, 'cachorros')) and 
            os.path.exists(os.path.join(caminho_validacao_dados, 'gatos')) and 
            len(os.listdir(os.path.join(caminho_treino_dados, 'cachorros'))) > 0):
        
        if not preparar_dataset_pre_organizado(caminho_base_dados):
            print('Não foi possível preparar o dataset. Verifique as pastas Cats e Dogs.')
    else:
        print('Pastas de treino e validação já estão prontas.')

    # Exemplo de uso dos geradores (apenas para teste)
    # gerador_treino, gerador_validacao = criar_geradores_de_dados(caminho_treino_dados, caminho_validacao_dados)
    # for dados_batch, rotulos_batch in gerador_treino:
    #     print(f'Shape do batch de treino: {dados_batch.shape}, Shape dos rótulos: {rotulos_batch.shape}')
    #     break
    # for dados_batch, rotulos_batch in gerador_validacao:
    #     print(f'Shape do batch de validação: {dados_batch.shape}, Shape dos rótulos: {rotulos_batch.shape}')
    #     break
