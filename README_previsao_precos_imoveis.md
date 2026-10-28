# Previsão de Preços Imobiliários com Regressão Linear Múltipla e Machine Learning

## Visão geral

Projeto de análise de dados e modelagem preditiva desenvolvido em Python para estimar o **valor mediano de imóveis em distritos da Califórnia**. O estudo utiliza dados geográficos, demográficos e socioeconômicos e compara modelos de **Regressão Linear Múltipla** e **Árvore de Decisão**.

O trabalho demonstra etapas relevantes de um fluxo de ciência de dados: exploração e visualização dos dados, identificação de valores ausentes, separação estratificada de amostras, preparação de variáveis, treinamento e avaliação de algoritmos de regressão.

## Objetivo

Investigar a relação entre características dos distritos e o valor mediano dos imóveis, preparando os dados e construindo modelos para estimar a variável-alvo `median_house_value`.

## Base de dados

- **Arquivo:** `housing.csv`
- **Observações:** 20.640
- **Colunas:** 10, incluindo a variável-alvo
- **Valores ausentes:** 207 registros na coluna `total_bedrooms`
- **Variável categórica:** `ocean_proximity`
- **Variável-alvo:** `median_house_value`

Entre os atributos utilizados estão localização (latitude e longitude), idade mediana dos imóveis, quantidade de cômodos e dormitórios, população, domicílios, renda mediana e proximidade com o oceano.

## Metodologia

1. **Exploração dos dados:** inspeção de tipos, estatísticas descritivas, histogramas, distribuições geográficas e relações entre atributos.
2. **Análise de correlação:** estudo das associações entre as variáveis numéricas e o valor mediano das residências, com atenção à renda mediana.
3. **Amostragem estratificada:** criação de faixas de renda e separação de **80% dos registros para treinamento** (16.512) e **20% para teste** (4.128), preservando a distribuição das categorias de renda.
4. **Pré-processamento:** imputação de valores numéricos ausentes pela mediana, padronização com `StandardScaler` e codificação de `ocean_proximity` com `OneHotEncoder`.
5. **Pipelines:** organização do tratamento numérico e categórico por meio de `Pipeline` e `ColumnTransformer` do Scikit-learn.
6. **Modelagem:** treinamento de `LinearRegression` e `DecisionTreeRegressor(max_depth=10)`.
7. **Avaliação:** cálculo de MAE, RMSE e R² e comparação dos resultados dos algoritmos.
8. **Análise estatística complementar:** ajuste de regressão por mínimos quadrados ordinários (`OLS`) utilizando Statsmodels para examinar coeficientes e outras estatísticas do modelo.

## Resultados

Os resultados abaixo são os **registrados no notebook e calculados sobre os dados de treinamento**:

| Métrica | Regressão Linear Múltipla | Árvore de Decisão |
|---|---:|---:|
| MAE | US$ 49.905,33 | US$ 32.067,27 |
| RMSE | US$ 69.050,56 | US$ 47.873,31 |
| R² | 0,6438 | 0,8288 |

No conjunto de treinamento, a Árvore de Decisão obteve **redução aproximada de 35,7% no MAE** em relação à Regressão Linear e aumento de aproximadamente **0,185 ponto no R²**.

**Importante:** o notebook reserva um conjunto de teste, mas não apresenta a avaliação final desses dois modelos nele. Os resultados descritos **não devem ser tratados como desempenho de generalização**; a avaliação no conjunto de teste ou por validação cruzada é uma melhoria recomendada.

## Tecnologias e bibliotecas

- **Python** — análise e modelagem
- **Pandas e NumPy** — manipulação e transformação de dados
- **Matplotlib** — gráficos exploratórios
- **Scikit-learn** — amostragem, imputação, codificação, escalonamento, pipelines, regressão, árvore de decisão e métricas
- **Statsmodels** — regressão OLS e análise dos coeficientes
- **Jupyter Notebook** — execução e documentação da análise

## Estrutura do repositório

```text
.
├── Regressao_Linear_Multipla.ipynb
├── housing.csv
└── README.md
```

## Como executar

1. Instale Python e abra o ambiente Jupyter Notebook ou JupyterLab.
2. Clone ou baixe o repositório, mantendo `housing.csv` na mesma pasta do notebook.
3. Instale as dependências:

   ```bash
   pip install pandas numpy matplotlib scikit-learn statsmodels notebook
   ```

4. Inicie o Jupyter:

   ```bash
   jupyter notebook
   ```

5. Abra `Regressao_Linear_Multipla.ipynb` e execute as células na ordem.

## Limitações e próximos passos

- **Avaliação fora da amostra:** calcular MAE, RMSE e R² no conjunto de teste, preservando-o para a avaliação final.
- **Validação cruzada:** comparar a estabilidade e o desempenho dos algoritmos em diferentes partições dos dados de treinamento.
- **Ajuste de hiperparâmetros:** avaliar configurações da Árvore de Decisão para reduzir o risco de sobreajuste.
- **Interpretação dos resultados:** investigar erros por faixa de preço e localização, incluindo o efeito de valores imobiliários limitados na base.
- **Reprodutibilidade:** fixar as versões das bibliotecas e revisar trechos que possam emitir avisos de compatibilidade com versões recentes do Pandas.

## Contexto do projeto

Estudo prático de regressão e preparação de dados com foco no desenvolvimento de competências em **Análise de Dados, Estatística e Machine Learning**. As métricas documentadas refletem exclusivamente as etapas executadas no notebook fornecido.
