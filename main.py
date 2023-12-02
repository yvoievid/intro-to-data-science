
import concurrent.futures
import subprocess

def run_flask():
    subprocess.run(['python', 'controllers.py'])

def run_pygame():
    subprocess.run(['python', 'simulation_start.py'])

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        flask_future = executor.submit(run_flask)
        pygame_future = executor.submit(run_pygame)

        # Wait for both processes to finish
        concurrent.futures.wait([flask_future, pygame_future], return_when=concurrent.futures.FIRST_COMPLETED)