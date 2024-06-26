# Repositório: Implementação de Gradiente Checkpoint para Grafos Gerais

Este repositório contém uma implementação do gradiente checkpoint para grafos gerais, projetado para otimizar o uso de memória durante o treinamento de modelos complexos. Aqui está uma visão geral dos principais arquivos e funcionalidades:

- **graph.py**: Implementa a estrutura do grafo, exigindo construção explícita através de suas funções dedicadas.

- **models.py**: Contém a implementação de dois modelos discutidos no artigo, essenciais para demonstrar a eficácia do método de gradiente checkpoint.

- **main.py**: Executa o modelo principal e foi usado para gerar as Figuras 5 e 6 do artigo. Para gerar os gráficos, o script inclui um comando que requer descomentários na linha 61 para executar apenas uma iteração do processo.

    Para executar o script, utilize o seguinte comando:
    ```bash
    python3 main.py --stats_path {path} --model {model} --segment_size {segment_size} --hidden_size {hidden_size}
    ```

- **time_x_memory.py**: Utilizado para analisar o uso de memória ao longo de uma iteração do modelo, conforme mostrado na Figura 6 do artigo.

    Para executar o script, utilize o comando
    ```bash
    python3 time_x_memory.py --time_path {path} --model {model} --segment_size {segment_size} --hidden_size {hidden_size} --log_interval {interval}
    ```
- **exec.py**: Script utilizado para automatizar a execução dos arquivos `main.py` e `time_x_memory.py` com as variações apresentadas no artigo.

- **images.ipynb**: Notebook Jupyter utilizado para gerar as imagens presentes no artigo, contribuindo para a visualização dos resultados obtidos.

Este repositório não só implementa o método de gradiente checkpoint para grafos gerais, mas também fornece as ferramentas necessárias para replicar os experimentos discutidos no artigo, facilitando a compreensão e validação dos resultados apresentados.
