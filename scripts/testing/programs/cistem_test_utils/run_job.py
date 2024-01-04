import os
import subprocess
import time


def run_job(temp_run_filename):
    start_time = time.time()
    subprocess.run(temp_run_filename)
    elapsed_time = time.time() - start_time
    os.remove(temp_run_filename)
    return elapsed_time
