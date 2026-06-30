# Classificação de Imagens: Cães e Gatos com Redes Neurais Convolucionais

**Aluno:** Fellipe Tavares  
**Disciplina:** Redes Neurais  
**Repositório:** https://github.com/FellpsTavares/Engenharia-de-Software-Redes-Neurais

---

## Objetivo

Classificação binária de imagens (gato ou cachorro) utilizando duas abordagens:

1. **CNN Personalizada** — arquitetura convolucional treinada do zero
2. **Transfer Learning com VGG16** — rede pré-treinada no ImageNet adaptada para a tarefa

**Dataset:** Cats vs Dogs — 25.000 imagens (12.500 por classe), divididas em 80% treino e 20% validação.

---

## Como Executar

### Opção 1 — Google Colab com GPU (recomendado)

> Ambiente utilizado no desenvolvimento: Google Colab com GPU T4.

1. Faça upload do arquivo `dataset_colab_full.zip` na raiz do Google Drive
2. Abra o notebook `colab_classificacao_pets.ipynb` no Google Colab
3. Ative a GPU: **Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4**
4. Execute todas as células: `Ctrl + F9`

O notebook realiza automaticamente:
- Montagem do Drive e extração do dataset
- Pré-processamento e data augmentation
- Treinamento da CNN com callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Geração de gráficos, matriz de confusão e relatório de classificação

### Opção 2 — Execução local (CPU)

```bash
pip install tensorflow scikit-learn matplotlib seaborn streamlit pillow
python projeto_classificacao_pet/main.py
```

### Opção 3 — Aplicação web interativa (Streamlit)

Permite fazer upload de uma imagem e visualizar a predição do modelo com a probabilidade calculada.

```bash
streamlit run projeto_classificacao_pet/app_streamlit.py
```

---

## Resultados Obtidos

### CNN Personalizada — concluída com sucesso

| Métrica | Valor |
|---------|-------|
| Melhor acurácia de validação | **94,87%** (epoch 17) |
| Loss de validação no melhor epoch | 0,1236 |
| Total de epochs executados | 20 |
| Parâmetros treináveis | 521.473 (~1,99 MB) |
| Tempo por epoch (GPU T4) | ~320 segundos |

**Evolução do treinamento:**

| Epoch | Acc Treino | Acc Validação | Evento |
|-------|-----------|---------------|--------|
| 1  | 60,4% | 62,3% | início |
| 6  | 81,9% | 87,0% | primeiro grande salto |
| 11 | 90,1% | 92,1% | ultrapassa 90% pela primeira vez |
| 14 | 91,5% | 83,9% | LR reduzido: 1e-3 → 5e-4 |
| 15 | 92,9% | 93,7% | recuperação pós redução de LR |
| **17** | **93,5%** | **94,9%** | **melhor resultado — modelo salvo** |
| 20 | 94,0% | 94,1% | LR reduzido: 5e-4 → 2,5e-4 |

A curva de validação acompanhou o treino sem divergência, indicando boa generalização sem overfitting severo.

### VGG16 Fine-tuned — não executado

O código está implementado e funcional no notebook, mas o treinamento não foi concluído por esgotamento do limite de uso gratuito do Google Colab (~5 horas de GPU por sessão). **Acurácia esperada com base na literatura: 96–98%.**

---

## Arquitetura da CNN Personalizada

```
Entrada (224×224×3)
    ├── Conv2D(32)  → BatchNormalization → MaxPooling2D
    ├── Conv2D(64)  → BatchNormalization → MaxPooling2D
    ├── Conv2D(128) → BatchNormalization → MaxPooling2D
    ├── Conv2D(256) → BatchNormalization → MaxPooling2D
    ├── GlobalAveragePooling2D
    ├── Dense(512, relu) → Dropout(0.5)
    └── Dense(1, sigmoid)   ← 0 = gato, 1 = cachorro
```

**Decisões de projeto:**
- **BatchNormalization** após cada convolução — estabiliza o treinamento e permite learning rates mais altos
- **GlobalAveragePooling2D** em vez de Flatten — reduz parâmetros de ~8M para ~520K, atuando também como regularizador
- **Dropout(0.5)** — desativa 50% dos neurônios aleatoriamente por iteração para evitar memorização

---

## Data Augmentation

| Técnica | Configuração |
|---------|-------------|
| Rotação | ±30° |
| Deslocamento horizontal/vertical | 20% |
| Zoom | 25% |
| Espelhamento horizontal | ativado |
| Variação de brilho | 75% a 125% |

---

## Callbacks de Treinamento

| Callback | Comportamento observado |
|----------|------------------------|
| `EarlyStopping` (patience=5) | Não ativou — modelo melhorou continuamente |
| `ReduceLROnPlateau` (factor=0.5) | Ativou no epoch 14 e no epoch 20 |
| `ModelCheckpoint` | Epoch 17 salvo com 94,87% de acurácia |

---

## Estrutura do Repositório

```
projeto_classificacao_pet/
├── colab_classificacao_pets.ipynb   ← Notebook principal (Google Colab + GPU)
├── main.py                          ← Script de treino local (CPU)
├── pre_processamento.py             ← Organização e carregamento do dataset
├── modelos.py                       ← Arquiteturas CNN e VGG16
├── treinamento_e_avaliacao.py       ← Treino, gráficos e métricas
├── app_streamlit.py                 ← Aplicação web interativa
└── saidas/                          ← Gráficos e relatórios gerados
```

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
