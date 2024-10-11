#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void dotProductKernel(float* a, float* b, float* result, int n) {
    float tmp = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        tmp = a[idx] * b[idx];
    }

    atomicAdd(result, tmp);
}

float dot_product_cuda(torch::Tensor a, torch::Tensor b) {
    int n = a.size(0);

    float* d_a;
    float* d_b;
    float* d_result;
    float h_result = 0.0f;

    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_a, a.data<float>(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data<float>(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    dotProductKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return h_result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_cuda", &dot_product_cuda, "Dot product using CUDA");
}
