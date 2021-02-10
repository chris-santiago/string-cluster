import pathlib

from setuptools import setup, find_packages

TITLE = 'string-cluster'
VERSION = '0.0.1'
DESCRIPTION = 'A SciKit-Learn style deduper.'
AUTHOR = 'Chris Santiago'
EMAIL = 'cjsantiago@gatech.edu'

HERE = pathlib.Path(__file__).absolute().parent
INSTALL_REQUIRES = HERE.joinpath('requirements.txt').read_text().split('\n')


def install_package():
    setup(
        name=TITLE,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        packages=find_packages(),
        install_require=INSTALL_REQUIRES,
        include_package_data=True
    )


if __name__ == '__main__':
    install_package()
