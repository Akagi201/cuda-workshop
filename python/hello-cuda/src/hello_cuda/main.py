import cupy as cp

PTX_SRC = r'''
  extern "C" __global__ void hello_cuda_from_gpu() {
    printf("GPU: Hello CUDA from GPU!\n");
  }
'''

if __name__ == "__main__":
  hello_cuda_from_gpu = cp.RawKernel(PTX_SRC, 'hello_cuda_from_gpu')
  hello_cuda_from_gpu((2,), (8,), ())
  cp.cuda.runtime.deviceSynchronize()
  print()
  print("CPU: Hello CUDA from CPU!")