### Multithreading
## When to use Multi Threading
###I/O-bound tasks: Tasks that spend more time waiting for I/O operations (e.g., file operations, network requests).
###  Concurrent execution: When you want to improve the throughput of your application by performing multiple operations concurrently.


import time
import threading


def print_num():
    for i in range(10):
        time.sleep(1)
        print(f"Number : {i}")


def print_str():
    for i in "Saraswati":
        time.sleep(1)
        print(f"Char : {i}")


# start = time.time()
# print_num()
# print_str()
# end = time.time()
# time_taken = end - start
# print(time_taken)



# create thread

t1 = threading.Thread(target=print_num)
t2 = threading.Thread(target=print_str)

start = time.time()
t1.start()
t2.start()
end = time.time()

t1.join()
t2.join()

time_taken = end - start
print(time_taken)

