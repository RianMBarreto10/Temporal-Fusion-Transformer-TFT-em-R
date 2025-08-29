# Temporal Fusion Transformer (TFT) em R

## 🚀 Visão Geral Este projeto traz para o R a primeira implementação completa do Temporal Fusion Transformer (TFT) — um modelo de deep learning de última geração para previsão de séries temporais, originalmente criado em Python e agora disponível para a comunidade R. O TFT combina previsões altamente precisas com transparência e interpretabilidade, permitindo não só prever o futuro, mas também entender quais variáveis realmente impactam os resultados e em que momentos.

## ✨ Principais Diferenciais

Seleção automática de variáveis: O TFT aprende, de forma dinâmica e em tempo real, quais variáveis são mais relevantes para cada previsão.
Explicabilidade: O modelo mostra quando e como cada variável influencia o resultado, trazendo clareza para decisões estratégicas.
Previsão probabilística (quantis): Gere intervalos de confiança e cenários, não apenas um valor pontual.
Pipeline completo: Da preparação dos dados à avaliação, tudo integrado e adaptável a diferentes contextos de negócio.
Pronto para aplicações reais: Suporta múltiplos tipos de dados, variáveis categóricas, features de calendário, lags, promoções e muito mais.
## 📦 Instalação

Instale o pacote torch para R (veja detalhes em https://torch.mlverse.org/start/installation/):
r
install.packages("torch")
    torch::install_torch()
Instale dependências adicionais:
r
install.packages(c("data.table", "dplyr", "lubridate"))
Clone este repositório e carregue o script principal:
r
# No R
    source("tft_completo.R")
## 🛠️ Como Usar

Exemplo rápido com dados simulados
# Execute o exemplo completo integrado
    resultado_completo <- exemplo_uso_tft_completo()
O pipeline irá:

- Simular dados realistas de vendas
- Preparar e normalizar as variáveis
- Treinar o TFT
- Gerar previsões probabilísticas
- Exibir métricas de desempenho

## Usando com seus próprios dados
1-Prepare seu data.frame com pelo menos uma coluna de data e a variável target.
2-Ajuste os nomes das colunas nos argumentos da função preparar_dados_vendas_completo().
3-Siga o fluxo:

    dados_preparados <- preparar_dados_vendas_completo(
      data = seu_dataframe,
      target_col = "nome_da_variavel_target",
      categorical_cols = c("coluna_cat1", "coluna_cat2"),
      date_col = "coluna_data"
      # ... outros argumentos
    )

    modelo <- temporal_fusion_transformer_enhanced(
      input_size = length(dados_preparados$feature_names$time_varying),
      # ... outros argumentos
    )

    resultado_treino <- treinar_tft_completo(modelo, dados_preparados)

    previsoes <- prever_tft_completo(modelo, dados_preparados)
## 📊 Interpretação e Explicabilidade O TFT permite extrair pesos de importância das variáveis ao longo do tempo. Você pode visualizar quais variáveis foram mais relevantes em cada previsão, facilitando a explicação dos resultados para áreas de negócio e tomada de decisão.

## 💡 Exemplos de Aplicação

Previsão de vendas e demanda em varejo
Planejamento de estoques
Previsão de consumo energético
Saúde (previsão de atendimentos, epidemias)
Qualquer problema de séries temporais com múltiplos fatores

## 🧠 Sobre o TFT O Temporal Fusion Transformer é um modelo híbrido que une o melhor das redes recorrentes (LSTMs), atenção multi-cabeça, seleção automática de variáveis e previsões probabilísticas.

Mais detalhes no paper original: https://arxiv.org/abs/1912.09363.

## 📝 Roadmap

- Disponibilizar como pacote CRAN
- Adicionar exemplos com datasets reais
- Suporte a múltiplas séries (multi-entity)
- Visualizações integradas de previsões e importâncias
- Interface mais amigável para usuários não técnicos

## 🤝 Contribua 
Sugestões, críticas e contribuições são muito bem-vindas! Abra uma issue ou faça um pull request.

Vamos juntos democratizar a previsão inteligente e interpretável no R!
