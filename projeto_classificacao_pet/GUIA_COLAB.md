# Guia de Execução — Google Colab

## Pré-requisitos concluídos
- Notebook aberto no Colab
- GPU T4 ativada (15 GB RAM GPU / 12.7 GB RAM sistema)
- Conta Kaggle com token gerado (`KGAT_...`)

---

## Como executar

### Opção rápida
Pressione `Ctrl+F9` (Executar tudo) e aguarde.

O notebook **pausará automaticamente na Célula 4** esperando o token Kaggle. Cole o token (`KGAT_...`) no campo que aparecer e pressione Enter. Tudo o mais roda sozinho.

### Opção manual (célula a célula)
Use `Shift+Enter` para executar uma célula e avançar para a próxima.
    
---

## O que cada célula faz

| Célula | O que faz | Tempo estimado |
|--------|-----------|----------------|
| 1 | Verifica GPU | segundos |
| 2 | Importa bibliotecas | segundos |
| 3 | Monta Google Drive — clique em **Permitir** | segundos |
| **4** | **Cola token Kaggle + baixa dataset** | ~2 min |
| 5 | Organiza pastas treino/validação | ~1 min |
| 6 | Cria geradores com data augmentation | segundos |
| 7 | Exibe exemplos do dataset | segundos |
| 8 | Define funções de avaliação | segundos |
| 9 | Define arquitetura CNN | segundos |
| 10 | **Treina CNN** (20 epochs máx, early stopping) | ~10–15 min |
| 11 | Avalia CNN — gráficos + matriz de confusão | ~1 min |
| 12 | **Treina VGG16 Fase 1** (base congelada) | ~10 min |
| 13 | **Treina VGG16 Fase 2** (fine-tuning block5) | ~10 min |
| 14 | Avalia VGG16 — gráficos + matriz de confusão | ~1 min |
| 15 | Comparação final dos dois modelos | segundos |
| 16 | Testa com foto própria (upload manual) | segundos |
| 17 | Lista arquivos salvos no Drive | segundos |

**Total estimado: 40–55 minutos** (sessão disponível: ~5 horas)

---

## O que será salvo no Google Drive

Pasta: `Meu Drive/classificacao_pets/`

```
classificacao_pets/
├── modelos/
│   ├── cnn_personalizada.keras       ← melhor checkpoint da CNN
│   └── vgg16_finetuned.keras         ← melhor checkpoint do VGG16
└── saidas/
    ├── exemplos_dataset.png          ← amostras com augmentation
    ├── historico_CNN_Personalizada.png
    ├── matriz_confusao_CNN_Personalizada.png
    ├── relatorio_CNN_Personalizada.txt
    ├── historico_VGG16_FineTuned.png
    ├── matriz_confusao_VGG16_FineTuned.png
    ├── relatorio_VGG16_FineTuned.txt
    └── comparacao_modelos.png
```

---

## Acurácia esperada

| Modelo | Acurácia esperada |
|--------|------------------|
| CNN Personalizada | 85–91% |
| VGG16 Fine-tuned | 92–96% |

---

## Problemas comuns

**"Você não tem unidades de computação disponíveis"**
Aguarde alguns minutos e tente novamente. A GPU gratuita do Colab tem fila de uso.

**Célula 4 — erro de autenticação Kaggle**
Verifique se o token foi copiado completo (começa com `KGAT_`). Gere um novo em kaggle.com > Settings > API > Create New Token.

**Sessão desconectada durante o treinamento**
Os modelos são salvos no Drive a cada epoch que melhora (`ModelCheckpoint`). Se reconectar, os arquivos `.keras` no Drive contêm o melhor estado até o momento.

**"CUDA out of memory"**
Reduza o `BATCH_SIZE` de 32 para 16 na Célula 6 e re-execute a partir daí.

---

## Melhorias implementadas vs. versão local

| Aspecto | Versão local (CPU) | Versão Colab (GPU) |
|---------|-------------------|-------------------|
| Hardware | CPU | T4 GPU (15 GB) |
| Tempo total | ~5–6 horas | ~45–55 minutos |
| Resolução das imagens | 150×150 | 224×224 (nativo VGG16) |
| Arquitetura CNN | 3 blocos, sem BatchNorm | 4 blocos + BatchNorm + GAP |
| VGG16 | Base 100% congelada | Fine-tuning block5 (fase 2) |
| Epochs máximos | 10 | 20 (com early stopping) |
| Acurácia esperada | ~80–85% | ~92–96% |
| Salvamento automático | Manual | Google Drive (melhor checkpoint) |
