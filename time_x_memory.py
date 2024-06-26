import psutil
import time
import threading
from main import train
import argparse

def log_memory_usage(model, segment_size, path, interval=0.01):
    process = psutil.Process()
    start_time = time.time()
    while True:
        mem_info = process.memory_info()
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(path, 'a+') as f:
            f.write(f"{model},{elapsed_time},{segment_size},{mem_info.rss / (1024 * 1024)}\n")
        time.sleep(interval)


if __name__ == "__main__":


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model and log parameters.')
    parser.add_argument('--time_path', type=str, help='Path to log status and parameters.')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--segment_size', type=int, help='Segment size for the model.')
    parser.add_argument('--hidden_size', type=int, help='Hidden size parameter.')
    parser.add_argument('--log_interval', type=float, help='Interval to log memory usage')

    args = parser.parse_args()

    # Start memory logging in a separate thread
    logging_thread = threading.Thread(target=log_memory_usage, args=((args.model, args.segment_size, args.time_path, args.log_interval)))
    logging_thread.daemon = True
    logging_thread.start()

    # Start training
    train(args.model, args.hidden_size, args.segment_size)


