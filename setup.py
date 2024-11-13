from setuptools import setup, find_packages

setup(
    name="analysis-utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'seaborn',
        'statsmodels',
        'scipy'
    ],
)