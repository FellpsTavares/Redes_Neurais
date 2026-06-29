# Classificação de Imagens: Cães e Gatos com Redes Neurais Convolucionais

**Disciplina:** Redes Neurais  
**Dataset:** Microsoft Cats vs Dogs (25.000 imagens)  
**Modelos:** CNN Personalizada + VGG16 com Transfer Learning  
**Ambiente:** Google Colab (GPU T4)

---

## Resultados Obtidos

### CNN Personalizada — treinamento concluído

| Métrica | Valor |
|---------|-------|
| Melhor val_accuracy (epoch 17/20) | **94,87%** |
| Epochs até convergência | 20 (com ReduceLROnPlateau) |
| Loss final (validação) | 0,1236 |
| Parâmetros treináveis | 521.473 (1,99 MB) |

**Evolução por epoch (destaques):**

| Epoch | Treino acc | Val acc | Evento |
|-------|------------|---------|--------|
| 1 | 60,4% | 62,3% | — |
| 6 | 81,9% | 87,0% | salto de generalização |
| 11 | 90,1% | 92,1% | primeira vez acima de 90% |
| 15 | 92,9% | 93,7% | LR reduzido → novo máximo |
| **17** | **93,5%** | **94,9%** | **melhor checkpoint salvo** |
| 20 | 94,0% | 94,1% | LR reduzido novamente |

O gráfico de acurácia mostra convergência saudável: validação acompanha o treino sem divergência significativa, indicando ausência de overfitting severo.

### VGG16 Fine-tuned — em andamento

Treinamento do VGG16 (2 fases) ainda não concluído. Acurácia esperada: **96–98%** com fine-tuning do block5.

---

## Objetivo

Desenvolver e comparar dois modelos de classificação binária de imagens — identificando se uma foto contém um **gato** ou um **cachorro** — utilizando técnicas de aprendizado profundo com TensorFlow/Keras.

---

## Estrutura do Projeto

```
projeto_classificacao_pet/
├── colab_classificacao_pets.ipynb   ← Notebook principal (Google Colab)
├── main.py                          ← Script de treino local (CPU)
├── pre_processamento.py             ← Carregamento e organização do dataset
├── modelos.py                       ← Definição das arquiteturas CNN e VGG16
├── treinamento_e_avaliacao.py       ← Treino, gráficos, métricas
├── app_streamlit.py                 ← Aplicação web interativa
├── GUIA_COLAB.md                    ← Guia passo a passo de execução
├── dados/                           ← Dataset organizado (não versionado)
│   ├── treino/
│   │   ├── gatos/       (10.000 imagens)
│   │   └── cachorros/   (10.000 imagens)
│   └── validacao/
│       ├── gatos/       (2.500 imagens)
│       └── cachorros/   (2.500 imagens)
├── modelos/                         ← Modelos .keras salvos (não versionados)
└── saidas/                          ← Gráficos e relatórios gerados
```

---

## Etapas de Desenvolvimento

### Etapa 1 — Preparação do Dataset

O dataset utilizado é o **Microsoft Cats vs Dogs** (versão original do desafio Kaggle), com 25.000 imagens totais.

Divisão:
- **80% treino** → 10.000 imagens por classe
- **20% validação** → 2.500 imagens por classe

### Etapa 2 — Data Augmentation

Transformações aplicadas durante o treino para aumentar artificialmente a diversidade dos dados:

| Técnica | Parâmetro | Efeito |
|---------|-----------|--------|
| Rotação | ±30° | Ângulos variados de câmera |
| Deslocamento | 20% | Centralização variada |
| Zoom | 25% | Distâncias diferentes |
| Espelhamento horizontal | Ativo | Dobra o dataset efetivamente |
| Variação de brilho | 75–125% | Diferentes iluminações |

### Etapa 3 — Modelo 1: CNN Personalizada

Arquitetura construída do zero com 4 blocos convolucionais progressivos:

```
Entrada (224×224×3)
    ├── Conv2D(32)  → BatchNorm → MaxPooling
    ├── Conv2D(64)  → BatchNorm → MaxPooling
    ├── Conv2D(128) → BatchNorm → MaxPooling
    ├── Conv2D(256) → BatchNorm → MaxPooling
    ├── GlobalAveragePooling2D
    ├── Dense(512, relu)
    ├── Dropout(0.5)
    └── Dense(1, sigmoid)
```

**Decisões técnicas:**
- **BatchNormalization** após cada convolução: estabiliza o gradiente, permite LR maior
- **GlobalAveragePooling2D** em vez de Flatten: reduz parâmetros de ~8M para ~520k
- **Dropout(0.5)**: previne memorização, força generalização

**Resultado real:** 94,87% de acurácia na validação (epoch 17)

### Etapa 4 — Modelo 2: Transfer Learning com VGG16

VGG16 pré-treinada no ImageNet (1,2M imagens, 1000 classes). Reutiliza features aprendidas e adapta apenas as camadas finais.

**Fase 1** (base congelada, LR=1e-3, 10 epochs):
- Treina apenas as camadas densas do topo
- Aproveita features do ImageNet diretamente

**Fase 2** (fine-tuning block5, LR=1e-5, 10 epochs):
- Descongela as últimas camadas convolucionais
- LR muito baixo para não destruir os pesos pré-treinados
- Especializa o modelo em cães e gatos (+3–5% de acurácia)

### Etapa 5 — Callbacks de Treinamento

| Callback | Função | Resultado observado |
|----------|--------|---------------------|
| `EarlyStopping` (patience=5) | Para se val_loss estagnar | Treino completou 20 epochs sem parar |
| `ReduceLROnPlateau` (factor=0.5) | Reduz LR quando estagna | Ativou na epoch 14 (1e-3→5e-4) e epoch 20 (5e-4→2.5e-4) |
| `ModelCheckpoint` | Salva melhor checkpoint | Epoch 17 salvo (94,87%) |

### Etapa 6 — Avaliação

Para cada modelo são gerados:
- Gráficos de acurácia e loss por epoch (treino vs validação)
- Matriz de confusão
- Relatório completo: precision, recall, F1-score por classe

**Nota técnica:** o gerador de validação usa `shuffle=False` para garantir alinhamento correto entre predições e rótulos verdadeiros durante a avaliação.

### Etapa 7 — Aplicação Web (Streamlit)

Interface para uso do modelo sem código:

```bash
streamlit run projeto_classificacao_pet/app_streamlit.py
```

---

## Como Executar no Google Colab

1. Coloque `dataset_colab_full.zip` (25.000 imagens) na raiz do Google Drive
2. Abra `colab_classificacao_pets.ipynb` no Colab
3. Ative GPU T4: **Ambiente de execução → Alterar tipo → T4 GPU**
4. Execute tudo: `Ctrl+F9`

Consulte [GUIA_COLAB.md](GUIA_COLAB.md) para detalhes célula a célula.

---

## Conceitos Aplicados

| Conceito | Onde é aplicado |
|----------|-----------------|
| CNN | Extração de features visuais (bordas, texturas, formas) |
| Transfer Learning | Reutilização dos pesos VGG16 (ImageNet) |
| Fine-tuning | Desbloqueio parcial da rede na Fase 2 |
| Data Augmentation | `ImageDataGenerator` com 6 transformações |
| BatchNormalization | Estabilização do gradiente por camada |
| Dropout | Regularização nas camadas densas |
| EarlyStopping | Parada automática quando não há melhora |
| ReduceLROnPlateau | Ajuste dinâmico do learning rate |
| ModelCheckpoint | Persistência do melhor estado do modelo |
| Matriz de Confusão | Análise de erros por classe |

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
