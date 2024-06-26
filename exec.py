import subprocess
from main import train
import argparse

def run_varieties():
    segments = [1, 2, 3, 4, 5]
    hidden_sizes = {'sequential': range(16, 257, 16), 'graph': range(1, 33)}
    models = ['graph', 'sequential']
    with open('./stats.csv', 'w') as f:
        f.write(f"model,segment_size,hidden_size,peak,time\n")
    for model in models:
        for segment_size in segments:
            for hidden_size in hidden_sizes[model]:
                # Define o comando a ser executado
                command = f"python3 main.py --stats_path ./stats.csv --model {model} --segment_size {segment_size} --hidden_size {hidden_size}"
                print(command)
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
                    break


def run_time_x_memory():
    segments = [1, 2, 3, 4, 5]
    hidden_sizes = {'sequential': [128], 'graph': [16]}
    models = ['graph', 'sequential']

    with open('./time.csv', 'w') as f:
        f.write(f"model,time,segment_size,memory\n")

    for model in models:
        for segment_size in segments:
            for hidden_size in hidden_sizes[model]:
                # Define o comando a ser executado
                command = f"python3 time_x_memory.py --time_path ./time.csv --model {model} --segment_size {segment_size} --hidden_size {hidden_size} --log_interval 0.1"

                # Executa o comando
                process = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Espera o processo terminar e captura a saída e os erros
                stdout, stderr = process.communicate()

                # Imprime a saída e os erros
                print(f"Saída:\n{stdout.decode()}")
                print(f"Erros:\n{stderr.decode()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model and log parameters.')
    parser.add_argument('--do', type=str, help='Each function call', choices=['varieties', 'time_x_memory'])

    if args.do == 'varieties':
        run_varieties()
    else :
        run_time_x_memory()
