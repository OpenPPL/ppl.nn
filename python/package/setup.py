from setuptools import setup
from setuptools.dist import Distribution

def GetVersionString():
    version_file = 'VERSION'
    return open(version_file, 'r').read()

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True
    def is_pure(self):
        return False

setup(
    name = 'pyppl',
    version = GetVersionString(),
    description = 'OpenPPL python APIs',
    author = 'OpenPPL',
    author_email = 'openppl.ai@hotmail.com',
    url = 'https://github.com/openppl-public',
    classifiers = [
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License, Version 2.0',
    ],
    python_requires = '>=3.6',
    packages = ['pyppl'],
    include_package_data = True,
    distclass = BinaryDistribution,
)
