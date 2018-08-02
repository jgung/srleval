from setuptools import setup

setup(
    name='srleval',
    version='0.1.0',
    description='CoNLL 2005 semantic role labeling task evaluation script, ported from Perl to Python',
    url='https://github.com/jgung/srleval',
    author='James Gung',
    author_email='james.gung@colorado.edu',
    license='MIT',
    packages=['srleval'],
    entry_points={
        'console_scripts': ['srleval=srleval.srleval:main'],
    }
)
