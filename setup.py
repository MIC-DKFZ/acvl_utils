from setuptools import setup, find_namespace_packages

setup(name='acvl_utils',
      packages=find_namespace_packages(include=["acvl_utils", "acvl_utils.*"]),
      version='0.2.1',
      description='Super cool utilities that we just love to use',
      # url='',
      author='Applied Computer Vision Lab, Helmholtz Imaging & Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz.de',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
          "numpy",
          "batchgenerators",
          "torch",
          "SimpleITK",
          "scikit-image",
          "connected-components-3d"
      ],
      )
