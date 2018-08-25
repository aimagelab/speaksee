from setuptools import setup, find_packages

setup(name='speaksee',
      version='0.0.dev1',
      description='PyTorch utilities and models for Visual-Semantic tasks',
      url='http://github.com/aimagelab/speaksee',
      author='Lorenzo Baraldi, Marcella Cornia',
      author_email='lorenzo.baraldi@unimore.it',
      packages=find_packages(exclude=('test', )),
      package_data={'speaksee': ['evaluation/stanford-corenlp-3.4.1.jar']},
      include_package_data=True,
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
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ],
      zip_safe=False)