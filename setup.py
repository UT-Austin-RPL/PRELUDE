import io
from setuptools import setup, find_packages

__version__ = '1.0.0'

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='prelude',
    version=__version__,
    author='anonymous authors',
    author_email='',
    description=('Python Environments for PRELUDE'),
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type='text/markdown',
    license="GPLv3",
    python_requires=">=3.6",
    keywords="PRELUDE (CoRL 2022)",
    packages=[package for package in find_packages()
                if package.startswith('a1sim')]
            +[package for package in find_packages(
                exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"])
              if package.startswith('tianshou')]
            +[package for package in find_packages() if package.startswith("robomimic")],
    tests_require=['pytest', 'mock'],
    include_package_data=True,
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "attrs==19.3.0",
        "pybullet>=3.0.7",
        "gym>=0.18.0",
        "numpy==1.19.5",
        "scipy>=1.6.2",
        "pybullet>=3.0.6",
        "typing==3.7.4.1",
        "numba>=0.51.0",
        "h5py>=2.10.0",
        "pyglet==1.5.0",
        "opencv-python>=4.5.2",
        "pynput",
        "inputs",
        "PyYAML",
        "python-dateutil",
        "gtimer",
        "hid",
        "tensorboard>=2.5.0",
        "tensorboardX",
        "psutil",
        "tqdm",
        "termcolor",
        "imageio",
        "imageio-ffmpeg",
        "egl_probe>=1.0.1"
        "wandb",
    ],
    extras_require={
        "dev": [
            "Sphinx",
            "sphinx_rtd_theme",
            "sphinxcontrib-bibtex",
            "flake8",
            "pytest",
            "pytest-cov",
            "ray>=1.0.0",
            "networkx",
            "mypy",
            "pydocstyle",
            "doc8",
        ],
        "atari": ["atari_py", "cv2"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    },
    zip_safe=False,
)
