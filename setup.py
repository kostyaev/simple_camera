from Cython.Distutils import build_ext
import setuptools
import numpy
import io
import os
import re

name = 'simple_camera'


def get_version():
    version_file = os.path.join(name, "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


setuptools.setup(
    name='simple_camera',
    version=get_version(),
    author="Dmitry Kostyaev",
    author_email="dm.kostyaev@gmail.com",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/kostyaev/simple_camera",
    cmdclass={'build_ext': build_ext},
    ext_modules=[setuptools.Extension("simple_camera.mesh_core_cython",
                                      sources=["simple_camera/lib/mesh_core_cython.pyx",
                                               "simple_camera/lib/mesh_core.cpp"],
                                      language='c++',
                                      include_dirs=[numpy.get_include()])],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['Cython', 'numpy'],
    packages=['simple_camera']
)
