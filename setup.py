# https://chat.openai.com/c/949daf38-8f6e-4e4a-b209-c3781099686c

from setuptools import setup, find_packages

setup(
    name="TabularWizard",
    version="0.1.4",
    packages=find_packages(),
    install_requires = [
        'contourpy',
        'cycler',
        'fonttools',
        'graphviz1',
        'joblib',
        'kiwisolver',
        'lightgbm',
        'catboost',
        'matplotlib',
        'numpy',
        'packaging',
        'pandas',
        'Pillow',
        'plotly',
        # 'pyaml',
        'pyparsing',
        'python-dateutil',
        'pytz1',
        'PyYAML',
        'scikit-learn',
        'scikit-optimize',
        'scipy',
        'seaborn',
        'six',
        'tenacity',
        'threadpoolctl',
        'tzdata',
        'xgboost'
        ]
        
     ,
    author="Zalman Goldstein",
    author_email="zaalgol@gmail.com",
    description="A brief description of TabularWizard.",
    keywords="tabular data ML",
)
