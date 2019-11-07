from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
import simple_camera

setup(
	name = 'simple_camera',
    version='0.2.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("simple_camera.mesh_core_cython",
                 sources=["simple_camera/lib/mesh_core_cython.pyx",
                          "simple_camera/lib/mesh_core.cpp"],
                 language='c++',
                 include_dirs=[numpy.get_include()])],
    packages=['simple_camera']
)
