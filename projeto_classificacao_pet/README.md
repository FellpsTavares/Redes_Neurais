# Classificação de Imagens: Cães e Gatos com Redes Neurais Convolucionais

**Disciplina:** Redes Neurais  
**Dataset:** Microsoft Cats vs Dogs (25.000 imagens)  
**Modelos:** CNN Personalizada + VGG16 com Transfer Learning  
**Ambiente de execução recomendado:** Google Colab (GPU T4)

---

## Objetivo

Desenvolver e comparar dois modelos de classificação binária de imagens — identificando se uma foto contém um **gato** ou um **cachorro** — utilizando técnicas de aprendizado profundo com TensorFlow/Keras.

O projeto explora duas abordagens distintas:
1. **CNN do zero** — arquitetura convolucional projetada e treinada inteiramente a partir dos dados
2. **Transfer Learning com VGG16** — reutilização de uma rede pré-treinada em 1,2 milhão de imagens (ImageNet), adaptada para a tarefa

---

## Estrutura do Projeto

```
projeto_classificacao_pet/
├── colab_classificacao_pets.ipynb   ← Notebook principal (rodar no Google Colab)
├── main.py                          ← Script de treino local (CPU)
├── pre_processamento.py             ← Carregamento e organização do dataset
├── modelos.py                       ← Definição das arquiteturas CNN e VGG16
├── treinamento_e_avaliacao.py       ← Treino, gráficos, métricas, matriz de confusão
├── app_streamlit.py                 ← Aplicação web interativa para predições
├── GUIA_COLAB.md                    ← Guia passo a passo de execução no Colab
├── dados/                           ← Dataset organizado (não versionado no Git)
│   ├── treino/
│   │   ├── gatos/
│   │   └── cachorros/
│   └── validacao/
│       ├── gatos/
│       └── cachorros/
├── modelos/                         ← Modelos treinados .keras (não versionados)
└── saidas/                          ← Gráficos e relatórios gerados
```

---

## Etapas de Desenvolvimento

### Etapa 1 — Preparação do Dataset

O dataset utilizado é o **Microsoft Cats vs Dogs** (versão original do desafio Kaggle), contendo **12.500 imagens de gatos** e **12.500 imagens de cachorros**, totalizando 25.000 imagens.

A organização segue a divisão:
- **80% para treino** → 10.000 imagens por classe
- **20% para validação** → 2.500 imagens por classe

```python
# pre_processamento.py
def preparar_dataset_pre_organizado(caminho_base, proporcao_validacao=0.2, caminho_origem=None):
    # Copia imagens para treino/validacao separando por classe
    # Usa shutil.copy2 (não move) para preservar o dataset original
```

### Etapa 2 — Data Augmentation

Para aumentar artificialmente a variedade dos dados de treino e reduzir overfitting, aplicamos transformações aleatórias em cada imagem durante o treinamento:

| Técnica | Parâmetro | Efeito |
|---------|-----------|--------|
| Rotação | ±30° | Simula fotos tiradas em ângulos diferentes |
| Deslocamento horizontal/vertical | 20% | Centralização variada do animal |
| Zoom | 25% | Diferentes distâncias da câmera |
| Espelhamento horizontal | Ativo | Dobra efetivamente o dataset |
| Variação de brilho | 75%–125% | Simula condições de iluminação diversas |

```python
gen_treino_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    ...
)
```

### Etapa 3 — Modelo 1: CNN Personalizada

Arquitetura convolucional construída do zero com 4 blocos convolucionais progressivos:

```
Entrada (224×224×3)
    │
    ├── Conv2D(32) → BatchNorm → MaxPooling
    ├── Conv2D(64) → BatchNorm → MaxPooling
    ├── Conv2D(128) → BatchNorm → MaxPooling
    ├── Conv2D(256) → BatchNorm → MaxPooling
    │
    ├── GlobalAveragePooling2D      ← substitui Flatten, menos parâmetros
    ├── Dense(512, relu)
    ├── Dropout(0.5)                ← previne overfitting
    └── Dense(1, sigmoid)           ← saída binária (0=gato, 1=cachorro)
```

**Decisões de arquitetura:**
- **BatchNormalization** após cada convolução: normaliza as ativações, tornando o treinamento mais estável e permitindo learning rates maiores
- **GlobalAveragePooling2D** em vez de Flatten: reduz drasticamente o número de parâmetros e atua como regularizador implícito
- **Dropout(0.5)** na camada densa: desativa aleatoriamente 50% dos neurônios por batch, forçando redundância e evitando memorização

### Etapa 4 — Modelo 2: Transfer Learning com VGG16

A VGG16 é uma rede com 16 camadas treinada no ImageNet. Em vez de treinar do zero, reutilizamos seus pesos (que já reconhecem bordas, texturas, formas) e adaptamos apenas as camadas finais para a nossa tarefa binária.

O treinamento ocorre em **duas fases**:

**Fase 1 — Base congelada (10 epochs, LR = 1e-3)**
```
VGG16 (pesos ImageNet, CONGELADA)
    │
    ├── GlobalAveragePooling2D
    ├── Dense(256, relu)
    ├── Dropout(0.5)
    └── Dense(1, sigmoid)          ← apenas estas camadas são treinadas
```

**Fase 2 — Fine-tuning do bloco 5 (10 epochs, LR = 1e-5)**
```
VGG16
    ├── block1–block4  (CONGELADOS)
    └── block5         (DESCONGELADO) ← retreinado com LR muito baixo
        │
        └── ... camadas densas
```

A Fase 2 permite que as últimas camadas convolucionais da VGG16 se especializem em características de cães e gatos, gerando o ganho extra de acurácia (+3–5%).

### Etapa 5 — Callbacks de Treinamento

Três mecanismos automáticos controlam o treinamento:

| Callback | Função |
|----------|--------|
| `EarlyStopping` | Para o treino quando a val_loss para de melhorar (evita treino desnecessário) |
| `ReduceLROnPlateau` | Reduz o learning rate pela metade quando o modelo estagna (sai de mínimos locais) |
| `ModelCheckpoint` | Salva automaticamente apenas o melhor checkpoint (maior val_accuracy) |

### Etapa 6 — Avaliação e Métricas

Para cada modelo são gerados:

- **Gráficos de acurácia e loss** por epoch (treino vs validação)
- **Matriz de confusão** — mostra quantos gatos/cachorros foram classificados corretamente e onde houve erro
- **Relatório de classificação completo** — precision, recall e F1-score por classe

### Etapa 7 — Aplicação Web (Streamlit)

Interface interativa para uso do modelo treinado sem necessidade de código:

```bash
streamlit run projeto_classificacao_pet/app_streamlit.py
```

Funcionalidades:
- Upload de qualquer imagem
- Predição em tempo real pelos dois modelos
- Exibição da probabilidade de ser gato ou cachorro
- Comparação visual dos dois resultados

---

## Como Executar

### Opção A — Google Colab (recomendado, GPU gratuita)

1. Abra o arquivo `colab_classificacao_pets.ipynb` no Google Colab
2. Vá em **Ambiente de execução → Alterar tipo → T4 GPU**
3. Execute todas as células com `Ctrl+F9`
4. O notebook baixa o dataset automaticamente, treina os dois modelos e salva tudo no Google Drive

Consulte o [GUIA_COLAB.md](GUIA_COLAB.md) para detalhes célula a célula.

### Opção B — Execução local (CPU, mais lento)

```bash
# 1. Instalar dependências
pip install tensorflow scikit-learn matplotlib seaborn streamlit pillow

# 2. Treinar os modelos
python projeto_classificacao_pet/main.py

# 3. Iniciar aplicação web
streamlit run projeto_classificacao_pet/app_streamlit.py
```

**Requisitos locais:** Python 3.9+, ~4 GB RAM, ~2 GB de espaço em disco

---

## Resultados Esperados

| Modelo | Acurácia esperada | Tempo de treino (GPU T4) |
|--------|-------------------|--------------------------|
| CNN Personalizada | 85–91% | ~10–15 min |
| VGG16 Fine-tuned | 92–97% | ~20–25 min |

A diferença de acurácia demonstra o valor do Transfer Learning: ao partir de uma rede já treinada em milhões de imagens, o VGG16 converge mais rápido e atinge maior acurácia mesmo com o mesmo dataset.

---

## Conceitos Aplicados

| Conceito | Onde é aplicado |
|----------|-----------------|
| Redes Neurais Convolucionais (CNN) | Extração de features visuais (bordas, texturas, formas) |
| Transfer Learning | Reutilização dos pesos do VGG16 treinado no ImageNet |
| Fine-tuning | Desbloqueio parcial da rede pré-treinada na Fase 2 |
| Data Augmentation | `ImageDataGenerator` com rotação, zoom, flip, brilho |
| Regularização | BatchNormalization + Dropout + EarlyStopping |
| Otimização adaptativa | Adam com ReduceLROnPlateau |
| Avaliação | Matriz de confusão, precision, recall, F1-score |

---

## Dependências

```
tensorflow >= 2.12
scikit-learn
matplotlib
seaborn
streamlit
pillow
```
