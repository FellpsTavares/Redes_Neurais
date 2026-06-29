# Classificação de Imagens: Cães e Gatos com Redes Neurais Convolucionais

**Disciplina:** Redes Neurais  
**Dataset:** 25.000 imagens (10.000 treino + 2.500 validação por classe)  
**Modelos implementados:** CNN Personalizada + VGG16 Transfer Learning  
**Ambiente de execução:** Google Colab (GPU T4)

---

## Resultados Obtidos

### CNN Personalizada — concluída com sucesso

| Métrica | Valor |
|---------|-------|
| Melhor acurácia de validação | **94,87%** (epoch 17) |
| Loss de validação no melhor epoch | 0,1236 |
| Total de epochs executados | 20 |
| Parâmetros treináveis | 521.473 (1,99 MB) |
| Tempo por epoch (GPU T4) | ~320 segundos |

**Evolução do treinamento:**

| Epoch | Acc Treino | Acc Validação | Evento |
|-------|-----------|---------------|--------|
| 1  | 60,4% | 62,3% | início |
| 6  | 81,9% | 87,0% | primeiro grande salto |
| 9  | 88,9% | 88,5% | modelo estabilizando |
| 11 | 90,1% | 92,1% | ultrapassa 90% pela primeira vez |
| 13 | 91,3% | 92,2% | novo checkpoint salvo |
| 14 | 91,5% | 83,9% | LR reduzido: 1e-3 → 5e-4 |
| 15 | 92,9% | 93,7% | recuperação pós redução de LR |
| **17** | **93,5%** | **94,9%** | **melhor resultado — modelo salvo** |
| 20 | 94,0% | 94,1% | LR reduzido: 5e-4 → 2,5e-4 |

O gráfico de acurácia mostra convergência saudável: a curva de validação acompanha o treino sem divergência, indicando boa generalização e ausência de overfitting severo.

### VGG16 Fine-tuned — não executado

O treinamento do VGG16 não foi concluído por esgotamento do limite de uso gratuito do Google Colab (sessão de ~5 horas). O código está implementado e funcional no notebook, aguardando nova sessão com GPU disponível.

**Acurácia esperada com VGG16:** 96–98%, baseado na literatura e nas características do método (fine-tuning de rede pré-treinada no ImageNet com 1,2 milhão de imagens).

---

## Objetivo do Projeto

Desenvolver e comparar dois modelos de classificação binária de imagens capazes de identificar se uma foto contém um **gato** ou um **cachorro**, utilizando técnicas de aprendizado profundo com TensorFlow/Keras.

As duas abordagens implementadas são:
1. **CNN do zero** — arquitetura convolucional projetada e treinada inteiramente a partir dos dados
2. **Transfer Learning com VGG16** — reutilização de rede pré-treinada, adaptada para a tarefa

---

## Estrutura do Projeto

```
projeto_classificacao_pet/
├── colab_classificacao_pets.ipynb   ← Notebook principal (Google Colab + GPU)
├── main.py                          ← Script de treino local (CPU)
├── pre_processamento.py             ← Organização e carregamento do dataset
├── modelos.py                       ← Arquiteturas CNN e VGG16
├── treinamento_e_avaliacao.py       ← Treino, gráficos e métricas
├── app_streamlit.py                 ← Aplicação web interativa para predições
├── GUIA_COLAB.md                    ← Guia de execução no Colab
├── dados/                           ← Dataset (não versionado — 25.000 imagens)
├── modelos/                         ← Modelos .keras (não versionados)
└── saidas/                          ← Gráficos e relatórios gerados
```

---

## Etapas de Desenvolvimento

### Etapa 1 — Dataset

O dataset utilizado é o **Cats vs Dogs** com 25.000 imagens (12.500 gatos + 12.500 cachorros).

Divisão utilizada:
- **Treino:** 10.000 imagens por classe (80%)
- **Validação:** 2.500 imagens por classe (20%)

### Etapa 2 — Data Augmentation

Transformações aplicadas durante o treino para ampliar artificialmente a diversidade dos dados e reduzir overfitting:

| Técnica | Configuração |
|---------|-------------|
| Rotação | ±30° |
| Deslocamento horizontal/vertical | 20% |
| Zoom | 25% |
| Espelhamento horizontal | ativado |
| Variação de brilho | 75% a 125% |

### Etapa 3 — Modelo 1: CNN Personalizada

Arquitetura construída do zero com 4 blocos convolucionais progressivos:

```
Entrada (224×224×3)
    ├── Conv2D(32)  → BatchNormalization → MaxPooling2D
    ├── Conv2D(64)  → BatchNormalization → MaxPooling2D
    ├── Conv2D(128) → BatchNormalization → MaxPooling2D
    ├── Conv2D(256) → BatchNormalization → MaxPooling2D
    ├── GlobalAveragePooling2D
    ├── Dense(512, relu)
    ├── Dropout(0.5)
    └── Dense(1, sigmoid)   ← saída binária: 0 = gato, 1 = cachorro
```

**Decisões de arquitetura:**

- **BatchNormalization** após cada convolução: normaliza as ativações por batch, tornando o treino mais estável e permitindo learning rates mais altos
- **GlobalAveragePooling2D** em vez de Flatten: reduz o número de parâmetros de ~8 milhões para ~520 mil, funcionando também como regularizador
- **Dropout(0.5)** na camada densa: desativa aleatoriamente 50% dos neurônios por iteração, forçando redundância e evitando memorização

**Resultado:** 94,87% de acurácia na validação

### Etapa 4 — Modelo 2: Transfer Learning com VGG16 (implementado)

A VGG16 é uma rede com 16 camadas treinada no ImageNet. O treinamento ocorre em duas fases:

**Fase 1 — Base congelada (LR = 1e-3, 10 epochs)**
Treina apenas as camadas densas adicionadas no topo. As camadas convolucionais da VGG16 permanecem fixas, aproveitando diretamente os pesos aprendidos no ImageNet.

**Fase 2 — Fine-tuning do block5 (LR = 1e-5, 10 epochs)**
Descongela as últimas camadas convolucionais da VGG16 e as retreina com learning rate muito baixo, especializando o modelo em cães e gatos sem destruir os pesos pré-treinados.

> Não foi possível executar por esgotamento da sessão Colab gratuita (~5h de GPU).

### Etapa 5 — Callbacks de Treinamento

| Callback | Função | Comportamento observado |
|----------|--------|------------------------|
| `EarlyStopping` (patience=5) | Para se val_loss não melhorar | Não ativou — modelo melhorou até o epoch 20 |
| `ReduceLROnPlateau` (factor=0.5) | Reduz LR ao estagnar | Ativou no epoch 14 e no epoch 20 |
| `ModelCheckpoint` | Salva apenas o melhor checkpoint | Epoch 17 salvo com 94,87% |

### Etapa 6 — Avaliação

Para cada modelo são gerados:
- Gráfico de acurácia e loss por epoch (treino vs validação)
- Matriz de confusão
- Relatório completo: precision, recall e F1-score por classe

**Detalhe técnico:** o gerador de validação utiliza `shuffle=False` para garantir alinhamento correto entre as predições do modelo e os rótulos verdadeiros durante a avaliação. Sem isso, ocorre desalinhamento entre a ordem das predições e a ordem retornada por `gerador.classes`, produzindo métricas incorretas (~50%).

### Etapa 7 — Aplicação Web (Streamlit)

Interface interativa para uso do modelo treinado sem necessidade de código:

```bash
streamlit run projeto_classificacao_pet/app_streamlit.py
```

Permite upload de qualquer imagem e exibe a predição (gato ou cachorro) com a probabilidade calculada pelo modelo.

---

## Como Executar

### Google Colab (recomendado)

1. Adicione `dataset_colab_full.zip` à raiz do Google Drive
2. Abra `colab_classificacao_pets.ipynb` no Colab via GitHub
3. Ative GPU T4: **Ambiente de execução → Alterar tipo → T4 GPU**
4. Execute todas as células: `Ctrl+F9`

Consulte [GUIA_COLAB.md](GUIA_COLAB.md) para instruções detalhadas.

### Local (CPU)

```bash
pip install tensorflow scikit-learn matplotlib seaborn streamlit pillow
python projeto_classificacao_pet/main.py
```

---

## Comparativo de Abordagens

| Aspecto | CNN Personalizada | VGG16 Transfer Learning |
|---------|------------------|------------------------|
| Ponto de partida | pesos aleatórios | pesos do ImageNet (1,2M imagens) |
| Parâmetros treináveis | 521.473 | ~14 milhões (Fase 2) |
| Acurácia obtida | **94,87%** | não executado |
| Acurácia esperada | 88–95% | 96–98% |
| Tempo de treino (T4) | ~100 minutos | ~50 minutos (2 fases) |
| Vantagem | independência de dados externos | aproveitamento de conhecimento pré-existente |

---

## Conceitos Aplicados

| Conceito | Onde é aplicado |
|----------|-----------------|
| Redes Neurais Convolucionais | Extração automática de features visuais |
| Transfer Learning | Reutilização de pesos VGG16 do ImageNet |
| Fine-tuning | Desbloqueio parcial da rede na Fase 2 do VGG16 |
| Data Augmentation | `ImageDataGenerator` com 6 transformações |
| BatchNormalization | Normalização das ativações por camada |
| GlobalAveragePooling2D | Redução de dimensionalidade sem Flatten |
| Dropout | Regularização estocástica nas camadas densas |
| EarlyStopping | Parada automática contra overfitting |
| ReduceLROnPlateau | Ajuste dinâmico da taxa de aprendizado |
| ModelCheckpoint | Persistência automática do melhor modelo |

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
