// Add with a single thread on GPU
#include <stdio.h>

__global__ void add(int a, int b, int *c) { *c = a + b; }

int main() {
  int c;      // host copies
  int *dev_c; // device copies
  int size = sizeof(int);

  // Allocate space on device
  cudaMalloc((void **)&dev_c, size);

  // Launch add() kernel on GPU
  add<<<1, 1>>>(8, 2, dev_c);

  // Copy result back to host
  cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

  printf("%d\n", c);

  // Cleanup
  cudaFree(dev_c);

  return 0;
}
