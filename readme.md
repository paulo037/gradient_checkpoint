# Repositório: Implementação de Gradiente Checkpoint para Grafos Gerais

Este repositório contém uma implementação do gradiente checkpoint para grafos gerais, projetado para otimizar o uso de memória durante o treinamento de modelos complexos. Aqui está uma visão geral dos principais arquivos e funcionalidades:

- **graph.py**: Implementa a estrutura do grafo, exigindo construção explícita através de suas funções dedicadas.

- **models.py**: Contém a implementação de dois modelos discutidos no artigo, essenciais para demonstrar a eficácia do método de gradiente checkpoint.

- **main.py**: Executa o modelo principal e foi usado para gerar as Figuras 5 e 6 do artigo. Para gerar os gráficos, o script inclui um comando de break na linha 61 para executar apenas uma iteração do processo. Caso queira efetivamente treinar o modelo, é necessário comentá-la

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

# Instruções para Reproduzir os Resultados

Para configurar e executar os experimentos descritos, siga as etapas abaixo:

1. **Instale as Dependências:**

   Execute o comando abaixo para instalar todas as dependências listadas no arquivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

2. **Execute os Experimentos de Variedades:**
    Para iniciar o experimento de variedades, execute o seguinte comando:
    ```bash
    python3 exec.py --do varieties
    ```

3. **Execute os Experimentos de Tempo vs Memória:**
    Para iniciar o experimento de análise de tempo versus memória, execute o comando abaixo:
    ```bash
    python3 exec.py --do time_x_memory
    ```
4. **Visualize os Resultados:**
    Abra e execute o notebook images.ipynb para visualizar os resultados dos experimentos. Você pode fazer isso utilizando um ambiente como o Jupyter Notebook ou Jupyter Lab.

5. **Personalize os Parâmetros dos Algoritmos:**
    Se desejar alterar os parâmetros dos algoritmos, edite o arquivo `exec.py` conforme necessário.