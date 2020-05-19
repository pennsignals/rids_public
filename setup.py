"""Setup rids."""

from os.path import join
from setuptools import (
    find_packages,
    setup,
)


CLASSIFIERS = (
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
)

INSTALL_REQUIRES = (
    'numpy',
    'pandas',
    'pymongo',
    'pymssql',
    'pyyaml',
    'requests',
    'scikit-learn',
    'twilio',
)

SETUP_REQUIRES = (
    'pytest-runner',
    'setuptools_scm',
)

TESTS_REQUIRE = (
    'flake8',
    'flake8-bugbear',
    'flake8-commas',
    'flake8-comprehensions',
    'flake8-docstrings',
    'flake8-logging-format',
    'flake8-mutable',
    'flake8-sorted-keys',
    'pep8-naming',
    'pluggy',
    'pylint',
    'pytest',
    'pytest-cov',
    'pytest-flake8',
    'pytest-mock',
    'pytest-pylint',
)


def long_description(file_name='README.md'):
    """Return long description."""
    with open(join(file_name), encoding='utf-8') as file:
        return file.read()


setup(
    name='rids',
    author='Penn Medicine Predictive Healthcare',
    classifiers=list(CLASSIFIERS),
    entry_points={
        'console_scripts': (
            (
                'rids.predict = '
                'rids.predict:Micro.main'),
            # (
            #     'rids.replay = '
            #     'rids.predict:Micro.replay'),
            (
                'rids.ping_predict = '
                'rids.predict:Micro.run_ping'),
            (
                'rids.notify = '
                'rids.notify:Micro.main'),
            (
                # 'rids.ping_notify =  rids.notify:Micro.run_ping'
                'rids.ping_notify = '
                'rids.notify:Micro.run_ping'),
        ),
    },
    extras_require={
        'tests': TESTS_REQUIRE,
    },
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    long_description=long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url='https://github.com/pennsignals/rids',
    use_scm_version=True,
)
