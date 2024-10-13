import torch
import torch.utils.cpp_extension as cpp_ext
import os

cuda_home = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'

# Проверяем, существует ли путь к CUDA
if not os.path.exists(cuda_home):
    raise RuntimeError(f"CUDA not found at {cuda_home}. Please check the installation path.")

# Указываем пути к заголовочным файлам и библиотекам
include_d = os.path.join(cuda_home, 'include')
library_d = os.path.join(cuda_home, 'lib/x64')

# Компиляция и загрузка расширения
cuda_extension = cpp_ext.load(
    name='cuda_extension',
    sources=['../dot_product_cuda.cpp'],
    extra_include_paths=[include_d],  # Список с путём к заголовочным файлам
    extra_ldflags=[f'-L{library_d}'],  # Флаги линковки для библиотек
    extra_cflags=['-gencode', 'arch=compute_60,code=sm_60'],  # Флаги для NVCC
    extra_cuda_cflags=['-ccbin', 'cl.exe'],  # Использование NVCC с C++
    verbose=True
)

# Пример использования CUDA-функции
a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
b = torch.tensor([4.0, 5.0, 6.0], device='cuda')

# Вычисление скалярного произведения
result = cuda_extension.dot_product_cuda(a, b)

print(f"Dot product: {result}")
