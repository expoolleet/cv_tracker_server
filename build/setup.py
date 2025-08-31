from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np
import platform

def get_compiler_flags():
    if platform.system() == "Windows":
        compile_args = [
            "/O2",
            "/Ot",
            "/arch:AVX2",
            "/fp:fast",
            "/openmp",
            "/GL",
        ]
        link_args = ["/LTCG"]
    else:
        compile_args = [
            "-O3",
            "-march=native",
            "-mtune=native",
            "-ffast-math",
            "-fopenmp",
            "-funroll-loops",
            "-ftree-vectorize",
            "-fPIC",
        ]
        link_args = ["-fopenmp"]
    return compile_args, link_args

compile_args, link_args = get_compiler_flags()

extensions = [
     Extension(
        name="synthetic_target",
        sources=["synthetic_target.pyx"],
        include_dirs=[np.get_include()],
        extra_link_args=link_args,
        extra_compile_args=compile_args,
            define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ]
    ),
    Extension(
        name="fast_mosse_tracker",
        sources=["fast_mosse_tracker.pyx"],
        include_dirs=[np.get_include()],
        extra_link_args=link_args,
        extra_compile_args=compile_args,
                define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ]
    ),
    Extension(
        name="xortracker",
        sources=["xortracker.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        }
    ),
    include_dirs=[np.get_include()]
)
