import multiprocessing
import random
import time
import numpy as np

def worker_function(data_chunk, result_queue):
    """Process a chunk of data and return the sum."""
    print(f"Worker processing data chunk: {data_chunk}")
    time.sleep(random.uniform(0.1, 0.5))  # Simulating a time-consuming task
    result = sum(data_chunk)
    print(f"Worker finished processing: {data_chunk}, result: {result}")
    result_queue.put(result)

def parallel_sum(data, num_workers):
    """Calculate the sum of data using multiple workers."""
    chunk_size = len(data) // num_workers
    processes = []
    result_queue = multiprocessing.Queue()
    
    for i in range(num_workers):
        start_index = i * chunk_size
        end_index = None if i == num_workers - 1 else (i + 1) * chunk_size
        data_chunk = data[start_index:end_index]
        p = multiprocessing.Process(target=worker_function, args=(data_chunk, result_queue))
        processes.append(p)

    for p in processes:
        p.start()
    
    results = []
    for _ in processes:
        results.append(result_queue.get())
    
    for p in processes:
        p.join()

    total_sum = sum(results)
    return total_sum

def generate_large_data(size):
    """Generate a large array of random numbers."""
    return np.random.randint(1, 100, size).tolist()

def main():
    total_data_size = 1000000
    num_workers = multiprocessing.cpu_count()  # Use available CPU cores for parallel processing
    data = generate_large_data(total_data_size)

    print(f"Starting parallel computation on {total_data_size} elements with {num_workers} workers...")
    start_time = time.time()
    total_sum = parallel_sum(data, num_workers)
    end_time = time.time()

    print(f"Total sum computed: {total_sum}")
    print(f"Time taken for parallel computation: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
    
def factorial(n):
    """Compute the factorial of a number recursively."""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def parallel_factorial(n, result_queue):
    """Worker function to compute factorial in parallel."""
    print(f"Calculating factorial of {n}")
    result = factorial(n)
    print(f"Factorial of {n} is {result}")
    result_queue.put(result)

def calculate_factorials_in_parallel(numbers):
    """Calculate factorials of a list of numbers in parallel."""
    processes = []
    result_queue = multiprocessing.Queue()
    
    for number in numbers:
        p = multiprocessing.Process(target=parallel_factorial, args=(number, result_queue))
        processes.append(p)

    for p in processes:
        p.start()
    
    results = []
    for _ in processes:
        results.append(result_queue.get())
    
    for p in processes:
        p.join()

    return results

def main_factorial():
    numbers_to_calculate = [5, 10, 15, 20, 25]
    print("Starting parallel factorial computation...")
    start_time = time.time()
    factorial_results = calculate_factorials_in_parallel(numbers_to_calculate)
    end_time = time.time()

    for number, result in zip(numbers_to_calculate, factorial_results):
        print(f"Factorial of {number} is {result}")

    print(f"Time taken for parallel factorial computation: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
    main_factorial()

def matrix_multiplication(A, B):
    """Multiplies two matrices A and B."""
    result = np.zeros((len(A), len(B[0])))
    for i in range(len(A)):
        for j in range(len(B[0])):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(len(B)))
    return result

def parallel_matrix_multiply_chunk(A_chunk, B, result_chunk, row_start, row_end):
    """Compute a chunk of the matrix multiplication."""
    for i in range(row_start, row_end):
        for j in range(len(B[0])):
            result_chunk[i - row_start][j] = sum(A_chunk[i - row_start][k] * B[k][j] for k in range(len(B)))

def parallel_matrix_multiply(A, B):
    """Multiply matrices A and B using parallel processing."""
    num_workers = multiprocessing.cpu_count()
    chunk_size = len(A) // num_workers
    processes = []
    result = np.zeros((len(A), len(B[0])))

    for i in range(num_workers):
        row_start = i * chunk_size
        row_end = None if i == num_workers - 1 else (i + 1) * chunk_size
        A_chunk = A[row_start:row_end]
        result_chunk = np.zeros((len(A_chunk), len(B[0])))
        p = multiprocessing.Process(target=parallel_matrix_multiply_chunk, args=(A_chunk, B, result_chunk, row_start, row_end))
        processes.append((p, result_chunk))
    
    for p, _ in processes:
        p.start()

    for index, (p, result_chunk) in enumerate(processes):
        p.join()
        result[index * chunk_size:(index + 1) * chunk_size] = result_chunk

    return result

def main_matrix_multiplication():
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)

    print("Starting parallel matrix multiplication...")
    start_time = time.time()
    result = parallel_matrix_multiply(A, B)
    end_time = time.time()

    print("Matrix multiplication completed.")
    print(f"Time taken for parallel matrix multiplication: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
    main_factorial()
    main_matrix_multiplication()

def generate_large_vector(size):
    """Generate a large vector as a NumPy array."""
    return np.random.rand(size)

def vector_addition(v1, v2):
    """Perform vector addition."""
    return v1 + v2

def parallel_vector_addition(v1, v2):
    """Add two vectors in parallel."""
    num_workers = multiprocessing.cpu_count()
    chunk_size = len(v1) // num_workers
    processes = []
    result_queue = multiprocessing.Queue()

    for i in range(num_workers):
        start_index = i * chunk_size
        end_index = None if i == num_workers - 1 else (i + 1) * chunk_size
        p = multiprocessing.Process(target=lambda q, v1_chunk, v2_chunk: q.put(vector_addition(v1_chunk, v2_chunk)), args=(result_queue, v1[start_index:end_index], v2[start_index:end_index]))
        processes.append(p)

    for p in processes:
        p.start()

    results = []
    for _ in processes:
        results.append(result_queue.get())

    for p in processes:
        p.join()

    return np.concatenate(results)

def main_vector_addition():
    v1 = generate_large_vector(1000000)
    v2 = generate_large_vector(1000000)

    print("Starting parallel vector addition...")
    start_time = time.time()
    result = parallel_vector_addition(v1, v2)
    end_time = time.time()

    print("Vector addition completed.")
    print(f"Time taken for parallel vector addition: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
    main_factorial()
    main_matrix_multiplication()
    main_vector_addition()

def parallel_sort(arr):
    """Sort an array in parallel."""
    if len(arr) < 2:
        return arr
    mid = len(arr) // 2
    left = multiprocessing.Process(target=parallel_sort, args=(arr[:mid],))
    right = multiprocessing.Process(target=parallel_sort, args=(arr[mid:],))
    left.start()
    right.start()
    left.join()
    right.join()
    return merge(parallel_sort(arr[:mid]), parallel_sort(arr[mid:]))

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    while left and right:
        if left[0] < right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left or right)
    return result

def main_parallel_sort():
    arr = np.random.randint(1, 100, 1000000).tolist()
    
    print("Starting parallel sort...")
    start_time = time.time()
    sorted_arr = parallel_sort(arr)
    end_time = time.time()

    print("Array sorting completed.")
    print(f"Time taken for parallel sorting: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
    main_factorial()
    main_matrix_multiplication()
    main_vector_addition()
    main_parallel_sort()