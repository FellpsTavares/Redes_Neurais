# Sistema de Classificação de Cães e Gatos

Este projeto implementa um sistema de classificação de imagens para distinguir entre cães e gatos, utilizando duas abordagens de redes neurais convolucionais: uma arquitetura CNN personalizada e um modelo baseado em Transfer Learning (VGG16). O sistema inclui pré-processamento de dados, treinamento de modelos, avaliação de desempenho com métricas e gráficos, e um painel interativo desenvolvido com Streamlit para visualização dos resultados.

## Estrutura do Projeto

```
projeto_classificacao_pet/
├── dados/                  # Contém o dataset (imagens de treino e validação)
│   ├── treino/             # Imagens para treinamento
│   │   ├── cachorros/
│   │   └── gatos/
│   └── validacao/          # Imagens para validação
│       ├── cachorros/
│       └── gatos/
├── modelos/                # Modelos treinados (.keras)
├── saidas/                 # Gráficos e relatórios de avaliação
├── pre_processamento.py   # Script para carregar, organizar e pré-processar o dataset
├── modelos.py              # Definição das arquiteturas dos modelos (CNN personalizada e Transfer Learning)
├── treinamento_e_avaliacao.py # Funções para treinamento, avaliação e geração de métricas
├── main.py                 # Script principal para orquestrar o treinamento e avaliação
└── app_streamlit.py        # Aplicação Streamlit para visualização interativa
└── README.md               # Este arquivo
```

## Requisitos

Para executar este projeto, você precisará ter Python 3.8+ e as seguintes bibliotecas instaladas:

- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`
- `pillow`

Você pode instalar todas as dependências usando pip:

```bash
sudo pip3 install tensorflow scikit-learn matplotlib seaborn streamlit pillow
```

## Como Usar

### 1. Download e Preparação do Dataset


### 2. Executar o Pré-processamento e Treinamento dos Modelos

O script `main.py` orquestra todo o processo de preparação do dataset, treinamento dos dois modelos (CNN personalizada e Transfer Learning) e a geração dos gráficos e relatórios de avaliação.

```bash
python3 projeto_classificacao_pet/main.py
```

Este script irá:
- Organizar o dataset em pastas de treino e validação (se ainda não estiver organizado).
- Criar geradores de dados com técnicas de *data augmentation*.
- Treinar a CNN personalizada.
- Treinar o modelo de Transfer Learning (VGG16).
- Salvar os modelos treinados na pasta `modelos/`.
- Gerar gráficos de accuracy/loss e matrizes de confusão na pasta `saidas/`.
- Gerar relatórios de classificação (precision, recall, f1-score) na pasta `saidas/`.

**Nota:** Para fins de demonstração e agilidade, o número de `epochs` no `main.py` foi definido como 2. Para obter melhores resultados, recomenda-se aumentar este valor (e.g., 10-50 epochs) ao usar o dataset completo.

### 3. Visualizar os Resultados com Streamlit

Após o treinamento, você pode iniciar o painel interativo do Streamlit para explorar os resultados e fazer predições em tempo real:

```bash
streamlit run projeto_classificacao_pet/app_streamlit.py
```

Ao executar este comando, uma nova aba será aberta no seu navegador com a aplicação Streamlit. Você poderá:
- Ver uma visão geral das arquiteturas dos modelos e estatísticas do dataset.
- Comparar os gráficos de treinamento e avaliação de ambos os modelos.
- Fazer upload de novas imagens para obter predições em tempo real.
- Acessar relatórios detalhados de classificação.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorias, correções de bugs ou novas funcionalidades.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes. (Nota: O arquivo LICENSE não está incluído neste exemplo, mas seria uma boa prática adicioná-lo.)
