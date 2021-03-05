from setuptools import setup, find_packages
import os
import shutil
import glob

for f in glob.glob("build/pygazebo/pygazebo*.so"):
    shutil.copy2(f, "python")

setup(
    name='pygazebo',
    version='0.0.1',
    install_requires=['numpy'], 
    package_dir={'': 'python'},
    packages=find_packages('python'),
    package_data={'pygazebo': ['pygazebo*.so']},
)
