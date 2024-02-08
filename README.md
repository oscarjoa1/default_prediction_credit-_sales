# Redução do número de inadimplências (default) nas vendas a crédito da empresa X-Health.

## Autor
Oscar J. O. Ayala

## Conteúdo

1. [**Definição e quadro geral do desafio**](#desafio) <br>
    1.1       [*Definição do desafio*](#desafio11)<br>
    1.2       [*Premissas do processo de venda*](#desafio12)<br>
      - 1.2.1 [*Perfil do cliente*](#desagio121)<br>
      - 1.2.2 [*Venda crédito  à vista: perfil do cliente, processo, prazo, inadimplência e impactos*](#desafio122)<br>
      - 1.2.3 [*Venda crédito  à vista: perfil do cliente, processo, prazo, inadimplência e impactos*](#desafio123)<br>    
      
   1.3 [*Classificação de crédito*](#desafio13)<br>
   1.4 [*Abordagem, sistema, tarefa e técnica*](#desafio14)<br>
   1.5 [*Fontes para aprimorar o ML*](#desafio14)<br>
   1.6 [*Limitações e Desafios de Dados*](#desafio16)<br>
   1.7 [*Aspectos Éticos e Regulatórios*](#desafio17)<br>
   1.8 [*Benefícios e riscos de ML*](#desafio18)<br>
2. [**Dados e Variáveis.**](#dados) <br>
3. [**Explore o Jupyter Notebook para realizar uma análise exploratória.**](#exploratoria) <br>
4. [**Explore o Jupyter Notebook  com o pipeline de estruturação do modelo probabilístico de inferência de default.**](#modelo) <br>
5. [**Explore o Jupyter Notebook  com função de predição.**](#predict) <br>
6. [**Recurso de modelos promissorios e modelo final**](#recurs) <br>

# <a id="desafio"></a>1) Definição e quadro geral do desafio
[*Back to top*](#top)

## <a id="desafio11"></a>1.1) Definição do desafio:  

- Produto oferecido: A empresa X-Health atua no comércio *B2B* vendendo dispositivos electrônicos voltado para saúde com amplo espectro de preços (versões mais acessíveis e mais caras), e de variada sofistificação/complexidade (algumas versoes mais simples e outras mais avançadas).

- Método de vendas: as vendas são feitas à crédito a vista ou em varias parcelas, segundo o combinado. 

- Problema: O time financeiro da empresa X-Health tem observado um número indesejábel de não-pagamentos. 

- Objetivo: Identificar clientes que podem resultar em não pagamentos, minimizando o númmero de não pagamentos.

## <a id="desafio12"></a>1.2) Premissas do processo de venda

### <a id="desafio121"></a>1.2.1) Perfil do cliente

O público-alvo principal são entidades jurídicas atuantes no setor da saúde, incluindo clínicas, hospitais, universidades e outros estabelecimentos similares, que buscam soluções personalizadas e/ou profissionais para otimizar suas operações produtivas. Embora não excluamos a possibilidade de atender pessoas físicas, é menos provável devido à necessidade de retorno sobre o investimento e conhecimento especializado exigido.
    
### <a id="desafio122"></a>1.2.2) Processo de venda crédito  à vista: 
   - Perfil teórico do cliente: pessoas que geralmente já possuem os recursos para pagamento em uma única vez. 
   - Fluxo de processo: i) Identificação do Cliente (interessados); ii) Negociação (definição do produto, determinação do preço e condições de pagamentos); iii) qualificação do cliente (em padrões rápida), iv) formalização de vendas (emissão de nota fiscal, contrato de compra e venda); iv) pagamento (detalhes do cartão e confirmação); v) entrega do produto (entrega imediata-recomendado, entrega agendada), vi) pós venda (atendimento ao cliente e garantia). 
   - Prazo padrão de recebimento: até 2 dias úteis.
   - Inadimplência: Embora pouco provável que acontezca, pode se dever a fraude do cartão, e problemas do produto. 
   - Impactos na operação: A empresa terá que arcar com o custo do produto ou serviço que não foi pago, a empresa pode ter que pagar taxas e multas e a imagem da empresa pode ser prejudicada. 
   
### <a id="desafio123"></a>1.2.3)  Processo de venda crédito  parcelado:
   - Perfil teórico do cliente: pessoas jurídicas que desejam preservação do capital de giro e manter o fluxo de caixa. 
   - Fluxo de processo: i) identificação do cliente (interessado),  ii) Negociação (definição do produto, determinação do preço e condições de pagamentos); iii) Qualificação do cliente (Avaliar a capacidade de pagamento do cliente, solicitar documentações para análise de crédito e aprovar ou negar o crédito.); iv) Condições de Pagamento (prazos para o cliente efeturar o pago); v) Entrega de Produtos ou Serviços; vi) Emissão de Fatura ou Documento de Cobrança; vii) pós venda (lembrete de pagamentos, atendimento ao cliente e garantia); ix) Registros de Pagamentos; x) Atualização do status default do cliente; xi) Compromisso finalizado.
   - Inadimplência: fraude, problemas financeiros e factores externos. 
   - Impactos na operação: perda de lucro, custos de cobrança, limitações na accesibilidade de crédito, perda do produto e custos de desenvolvimento do produto. Fatores externos como mudanças na situação econômica ou eventos inesperados, podem influenciar a inadimplência e não serem capturados pelo modelo.

### <a id="desafio13"></a>1.3) Classificação de crédito

- A avaliação de crédito é resumida por duas métricas essenciais: a Probabilidade de Inadimplência (PD) e a Perda Esperada em caso de Inadimplência (LGD).

- Neste contexto, focamos na PD, que indica a probabilidade de falha no pagamento do empréstimo. Essa métrica está vinculada à capacidade de geração de renda do mutuário durante o ciclo de vida do empréstimo.

##  <a id="desafio14"></a>1.4) Abordagem, sistema, tarefa e técnica.

- **Usam-se modelos de ML ou econométricos?** Optamos por modelos de ML, focalizando a melhor previsão fora da amostra da variável de resultado.

- **Que tipo de treinamento será necessário para o modelo?** Será necessário um sistema de ML supervisionado, uma vez que cada instância ou registro possui um label de *default*.

- **Qual será a tarefa desempenhada pelo modelo?** A tarefa desempenhada pelo modelo é uma classificação, dado que o resultado padrão é binário ("default" ou "não default").

- **Deve-se utilizar técnicas de aprendizado em lote ou em tempo real?** Recomenda-se o uso de técnicas de aprendizado em lote, uma vez que a classificação dos clientes não muda rapidamente.

##  <a id="desafio15"></a>1.5) Fontes para aprimorar o ML 

Aqui se aplicam os cinco Cs do crédito, aplicados principalmente a mutuários comerciais, têm atributos comuns à classificação de crédito pessoal. O aprendizado de máquina é essencial para aprimorar a avaliação de risco de crédito. Destaco aspectos-chave:

- **Capacidade:** Refere-se à capacidade financeira do mutuário, avaliando sua capacidade de pagamento considerando obrigações e lucratividade.

- **Estrutura de Capital:** Medida pela proporção de capital próprio em relação aos passivos totais. Maior participação de capital próprio indica maior financiamento pelos proprietários.

- **Cobertura:** Importante em empréstimos com garantia, onde ativos tangíveis oferecidos como garantia representam a cobertura.

- **Caráter:** Reflete o histórico de pagamentos, inadimplência, fraude e outros fatores que indicam a confiabilidade do mutuário. O Score de Crédito resume essas informações.

- **Condições:** Elementos macroeconômicos e fatores externos que influenciam o desempenho do empréstimo, como taxa de crescimento, desemprego, inflação, regulamentações e fatores setoriais e geográficos.

##  <a id="desafio16"></a>1.6) Limitações e Desafios de Dados: 

- Caso ocorrer a presença de uma classe de mutuários historicamente negados crédito por motivos não comerciais (como gênero, raça, religião, etnia, residência em determinadas áreas, nacionalidade, etc.) pode resultar em discriminação contra clientes desses grupos, causando viés no modelo. Espera-se que se tenha feito a exclusão dessas variáveis do conjunto de dados.

- Se a amostra sub-representa ou exclui uma atividade principal ou tipo de sociedade específica, o modelo pode não capturar padrões e relações relevantes durante o treinamento.

- A falta de atualização dos dados pode impedir o modelo de capturar mudanças ou tendências recentes.

- A presença de uma classe de mutuários historicamente negados crédito por motivos não comerciais (como gênero, raça, religião, etnia, residência em determinadas áreas, nacionalidade, etc.) pode resultar em discriminação contra clientes desses grupos, causando viés no modelo. Espera-se que se tenha feito a exclusão dessas variáveis do conjunto de dados.

## <a id="desafio17"></a>1.7)  Aspectos Éticos e Regulatórios: 

- A opacidade dos modelos de crédito baseados em aprendizado de máquina apresenta riscos para a proteção do consumidor.

- Há o risco de clientes falsificarem indicadores para obter uma avaliação melhor, o que pode prejudicar a pontuação baseada em aprendizado de máquina.

- O uso de informações de grandes conjuntos de dados obtida de agentes externos de orgãos regulares, não só levanta preocupações sobre a relevância (ruído) das informações, mas também suscita questões de proteção do consumidor, pois a exclusão pode ocorrer com base em decisões de computador inexplicáveis.

##  <a id="desafio18"></a>1.8) Benefícios e riscos de ML 

- **Benefícios**: avaliação mais rápida e econômica para a empresa, previsão aprimorada da inadimplência, decisões mais eficientes, e aumento da competitividade.

- **Riscos**: exclusão injusta de clientes qualificados por discriminação, falta de interpretabilidade em modelos de ML, exclusão devido a decisões computacionais inexplicáveis, preocupações com segurança e tratamento de dados.

# <a id="dados"></a> 2) Dados e Variáveis

Os dados são propriedade do Kognita Lab e possuem as seguintes características:
- Caminho: o dataset disponibilizado pela X-Health encontra-se em
  `./_data_set/dataset_2021-5-26-10-14.csv`
- Estrutura do .csv: para ler o arquivo, use sep = '\t' e encoding = 'utf-8'.
- O dataset possui tanto variáveis internas (decorrentes do comportamento histórico do cliente B2B junto à X-Health), quanto variáveis
externas consultadas em bureaus de crédito, como o Serasa.
- Cada linha do dataset representa um evento de compra de um conjunto de produtos, e tanto as variáveis internas quanto externas
- representam uma fotografia do cliente naquele instante.
- Valores faltantes estão indicados no dataset como *missing*.
  
Dicionário de dados:

| nome_coluna | desc |
| :---: | :---: |
| default_3months | Quantidade de default nos últimos 3 meses |
| ioi_Xmonths | Intervalo médio entre pedidos (em dias) nos últimos X meses |
| valor_por_vencer | Total em pagamentos a vencer do cliente B2B, em Reais |
| valor_vencido | Total em pagamentos vencidos do cliente B2B, em Reais |
| valor_quitado | Total (em Reais) pago no histórico de compras do cliente B2B |
| quant_protestos | Quantidade de protestos de títulos de pagamento apresentados no Serasa |
| valor_protestos | Valor total (em Reais) dos protestos de títulos de pagamento apresentados no Serasa |
| quant_acao_judicial | Quantidade de açôes judiciais apresentadas pelo Serasa |
| acao_judicial_valor | Valor total das açōes judiciais (Serasa) |
| participacao_falencia_valor | Valor total (em Reais) de falências apresentadas pelo Serasa |
| dividas_vencidas_valor | Valor total de dividas vencidas (Serasa) |
| dividas_vencidas_qtd | Quantidade total de dividas vencidas (Serasa) |
| falencia_concordata_qtd | Quantidade de concordatas (Serasa) |
| tipo_sociedade | Tipo de sociedade do cliente B2B |
| opcao_tributaria | Opçāo tributária do cliente B2B |
| atividade_principal | Atividade principal do cliente B2B |
| forma_pagamento | Forma de pagamento combinada para o pedido |
| valor_total_pedido | Valor total (em Reais) do pedido em questão |
| month | Mès do pedido |
| year | Ano do pedido |
| default | Status do pedido: default = 0 (pago em dia), default = 1 (pagamento näo-realizado, calote concretizado) |

# <a id="exploratoria"></a> 3) Explore o Jupyter Notebook para realizar uma análise exploratória
`./_result/analise_exploratoria.ipynd`

# <a id="modelo"></a> 4) Explore o Jupyter Notebook  com o pipeline de estruturação do modelo probabilístico de inferência de default
`./_result/pipeline_modelo_inferencia_default.ipynd`

# <a id="predict"></a> 5) Explore o Jupyter Notebook  com função de predição
`./_result/funcao_previsao.ipynd`

# <a id="recurs"></a> 6) Recurso de modelos promissorios e modelo final
`./_data_set`



> ##  Muito obrigado por estar comigo. 
