#include <cstdio>

__global__ void hello_cuda_from_gpu() {
    printf("GPU: Hello, CUDA!\n");
}

void hello_cuda_from_cpu() {
    printf("CPU: Hello, CUDA!\n");
}

int main() {
    {
        // GPU设备端核函数所指定的总线程数 = gridDim * blockDim
        const int GRID_DIM = 2;          // grid网格大小（线程块数量）
        const int BLOCK_DIM = 8;         // block线程块大小（每个线程块中的线程数量）
        hello_cuda_from_gpu<<<GRID_DIM, BLOCK_DIM>>>();  // GPU设备端核函数调用
        cudaDeviceSynchronize();        // 同步CPU主机端和GPU设备端
    }
    printf("\n");
    {
        hello_cuda_from_cpu();
    }
    return 0;
}