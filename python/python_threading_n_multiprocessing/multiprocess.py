## Processes that run in parallel
### CPU-Bound Tasks-Tasks that are heavy on CPU usage (e.g., mathematical computations, data processing).
## Parallel execution- Multiple cores of the CPU

import time
from multiprocessing import Process

def square_numbers(num):
    for i in range(num):
        time.sleep(1)
        print(f"Square: {i*i}")

def cube_numbers(num):
    for i in range(num):
        time.sleep(1.5)
        print(f"Cube: {i * i * i}")

if __name__=="__main__":

    ## create 2 processes
    p1=Process(target=square_numbers,args=(5,))
    p2=Process(target=cube_numbers,args=(5,))
    
    start = time.time()
    ## start the process
    p1.start()
    p2.start()
    ## Wait for the process to complete
    p1.join()
    p2.join()

    finished_time=time.time()-start
    print(finished_time)