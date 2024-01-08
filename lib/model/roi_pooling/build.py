# from __future__ import print_function
# import os
# import torch
# from torch.utils.ffi import create_extension


# sources = ['src/roi_pooling.c']
# headers = ['src/roi_pooling.h']
# extra_objects = []
# defines = []
# with_cuda = False

# this_file = os.path.dirname(os.path.realpath(__file__))
# print(this_file)

# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['src/roi_pooling_cuda.c']
#     headers += ['src/roi_pooling_cuda.h']
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True
#     extra_objects = ['src/roi_pooling.cu.o']
#     extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# ffi = create_extension(
#     '_ext.roi_pooling',
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects
# )

# if __name__ == '__main__':
#     ffi.build()

from __future__ import print_function
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

sources = ['src/roi_pooling.c']
headers = ['src/roi_pooling.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_pooling_cuda.c']
    headers += ['src/roi_pooling_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    extra_objects = ['src/roi_pooling_kernel.cu.o']  # Adjusted file name
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ext_modules = [
    CUDAExtension(
        name='_ext.roi_pooling',
        sources=sources + extra_objects,
        include_dirs=[os.path.dirname(this_file)],
        define_macros=defines,
        extra_compile_args={'cxx': [], 'nvcc': []}
    )
] if with_cuda else []

setup(
    name='roi_pooling_extension',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)

if __name__ == '__main__':
    setup()

