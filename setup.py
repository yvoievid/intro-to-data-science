from os import path

import setuptools
from setuptools import setup

extras = {
    'test': ['pytest', 'pytest_cases'],
    'develop': ['imageio'],
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='operation_simulation',
      version='0.0.14',
      long_description_content_type='text/markdown',
      long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
      license='MIT License',
      packages=setuptools.find_packages(),
      install_requires=[
          'scipy>=1.3.0',
          'numpy>=1.16.4',
          'pyglet>=1.4.0,<=1.5.27',
          'cloudpickle==2.0.0',
          'gym>=0.19.0,<=0.20.0',
          'pillow>=7.2.0',
          'six>=1.16.0',
          'opencv-python>=4.8.0'
      ],
      extras_require=extras,
      tests_require=extras['test'],
      python_requires='>=3.6, <3.12',
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      )
