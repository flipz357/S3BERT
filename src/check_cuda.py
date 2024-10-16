import torch

def check_cuda_and_gpus():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        # Print details for each GPU
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # Convert bytes to MB
            print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} MB")
    else:
        print("No CUDA-enabled GPU found.")

if __name__ == "__main__":
    check_cuda_and_gpus()