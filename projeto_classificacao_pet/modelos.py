
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16

def criar_cnn_personalizada(input_shape=(150, 150, 3)):
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

def criar_modelo_transfer_learning(input_shape=(150, 150, 3)):
    # Carregar o modelo VGG16 pré-treinado no ImageNet, sem as camadas densas do topo
    modelo_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar as camadas do modelo base para não serem treinadas
    for layer in modelo_base.layers:
        layer.trainable = False

    # Adicionar camadas densas personalizadas no topo do modelo base
    entrada = Input(shape=input_shape)
    x = modelo_base(entrada, training=False) # Garante que o modelo base roda em modo inferência
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    saida = Dense(1, activation='sigmoid')(x)

    modelo = Model(inputs=entrada, outputs=saida)
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

if __name__ == '__main__':
    print('Testando CNN personalizada...')
    cnn_modelo = criar_cnn_personalizada()
    cnn_modelo.summary()

    print('\nTestando modelo de Transfer Learning (VGG16)...')
    tl_modelo = criar_modelo_transfer_learning()
    tl_modelo.summary()
