import subprocess

segments = [1, 2, 3, 4, 5]
hidden_sizes = {'sequential': range(16, 257, 16), 'graph': range(1, 32)}
models = ['graph', 'sequential']


for model in models:
    for segment_size in segments:
        for hidden_size in hidden_sizes[model]:
            # Define o comando a ser executado
            command = f"python3 main.py --stats_path ./stats.csv --model {model} --segment_size {segment_size} --hidden_size {hidden_size}"

            # Executa o comando
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Espera o processo terminar e captura a saída e os erros
            stdout, stderr = process.communicate()

            # Imprime a saída e os erros
            print(f"Saída:\n{stdout.decode()}")
            print(f"Erros:\n{stderr.decode()}")
            if process.returncode != 0:
                with open('./stats.csv', 'a+') as f:
                    f.write(f"{model},{segment_size},{hidden_size},OM,0\n")
