from setuptools import setup, find_packages

setup(name='pyxlma',
    version=0.1,
    description='VHF LMA post-processing and visualization',
    packages=find_packages(),
    url='https://github.com/deeplycloudy/xlma-python/',
    long_description=open('README.md').read(),
    include_package_data=True,
    )