import torch
import torch.utils.cpp_extension as cpp_ext
import os

source_file = '../dot_product_cuda.cpp'

cuda_extension = cpp_ext.load(
    name = 'cuda_extension',
    sources = [source_file],
    verbose= True
)
a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
b = torch.tensor([4.0, 5.0, 6.0], device='cuda')

# Вызов CUDA-функции для вычисления скалярного произведения
result = cuda_extension.dot_product_cuda(a, b)

print(f"Dot product: {result}")