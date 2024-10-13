from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='dot_product_cuda',
    ext_modules=[
        CUDAExtension(
            name='dot_product_cuda',  # Имя модуля
            sources=['../dot_product_cuda.cpp'],  # Источник - наш .cu файл
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)