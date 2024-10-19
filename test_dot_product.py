import os
import torch
import torch.utils.cpp_extension as cpp_extension
import unittest
import math

# Установите архитектуру CUDA, если она ещё не настроена
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.0'

# Компиляция CUDA-расширения
dot_product_cuda = cpp_extension.load(
    name='dot_product_cuda',
    sources=['dot_product_cuda.cu'],
    extra_cuda_cflags=['-gencode', 'arch=compute_75,code=sm_75']
)

class TestDotProductCuda(unittest.TestCase):
    def __init__(self, num_tests=1, *args, **kwargs):
        super(TestDotProductCuda, self).__init__(*args, **kwargs)
        self.num_tests = num_tests

    def runTest(self):
        results = []
        for _ in range(self.num_tests):
            a = torch.randn(10, device='cuda')
            b = torch.randn(10, device='cuda')

            # Вычисление с помощью расширения
            cuda_result = dot_product_cuda.dot_product_cuda(a, b)

            # Вычисление с помощью стандартного метода PyTorch
            torch_result = torch.dot(a, b).item()

            # Проверка совпадения результатов с приемлемой точностью
            self.assertTrue(math.isclose(cuda_result, torch_result, rel_tol=1e-6))
            results.append((cuda_result, torch_result))

        for i, (cuda_result, torch_result) in enumerate(results):
            print(f"Test {i + 1}: CUDA result = {cuda_result}, PyTorch result = {torch_result}")

if __name__ == '__main__':
    # Ввод количества тестов
    num_tests = int(input("Введите количество тестов: "))  # Ввод количества тестов
    # Создаем тестовый набор
    suite = unittest.TestSuite()
    suite.addTest(TestDotProductCuda(num_tests=num_tests))
    unittest.TextTestRunner().run(suite)
