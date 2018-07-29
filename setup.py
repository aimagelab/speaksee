from setuptools import setup, find_packages

setup(name='speaksee',
      version='0.1',
      description='Captioning and cross-modal retrieval algorithms',
      url='http://github.com/aimagelab/speaksee',
      author='Lorenzo Baraldi, Marcella Cornia',
      author_email='lorenzo.baraldi@unimore.it',
      license='MIT',
      packages=find_packages(exclude=('test', )),
      install_requires=[
          'torch>=0.4.0',
          'torchvision',
          'numpy',
          'h5py',
          'tqdm',
          'requests',
          'pycocotools',
          'matplotlib',
          'pathos',
      ],
      zip_safe=False)