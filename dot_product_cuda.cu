#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA ядро для вычисления скалярного произведения
__global__ void dotProductKernel(float* a, float* b, float* result, int n) {
    __shared__ float cache[256]; // Локальная память для каждого блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    if (idx < n) {
        temp = a[idx] * b[idx];
    }
    
    // Сохраняем результат в shared memory
    cache[cacheIdx] = temp;
    __syncthreads();

    // Редукция внутри блока
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Атомарное сложение результатов блоков
    if (cacheIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}

float dot_product_cuda(torch::Tensor a, torch::Tensor b) {
    int n = a.size(0);

    // Указатели на данные на устройстве
    float* d_a;
    float* d_b;
    float* d_result;
    float h_result = 0.0f;

    // Выделяем память на устройстве
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    // Копируем данные с хоста на устройство
    cudaMemcpy(d_a, a.data_ptr<float>(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data_ptr<float>(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Запускаем CUDA ядро
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    dotProductKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, n);

    // Синхронизируем выполнение
    cudaDeviceSynchronize();

    // Копируем результат обратно на хост
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождаем память на устройстве
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return h_result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_cuda", &dot_product_cuda, "Dot product using CUDA");
}