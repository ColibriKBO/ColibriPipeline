from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('colibri_primary_filter.pyx',annotate=True),include_dirs=[numpy.get_include()])
