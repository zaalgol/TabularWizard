# https://chat.openai.com/c/949daf38-8f6e-4e4a-b209-c3781099686c

from setuptools import setup, find_packages

setup(
    name="TabularWizard",
    version="0.1.1",
    packages=find_packages(),
    install_requires = [
        'catboost==1.2.2',
        'contourpy==1.1.1',
        'cycler==0.12.0',
        'fonttools==4.43.0',
        'graphviz==0.20.1',
        'joblib==1.3.2',
        'kiwisolver==1.4.5',
        'lightgbm==4.1.0',
        'matplotlib==3.8.0',
        'numpy==1.26.3',
        'packaging==23.1',
        'pandas==2.2.0',
        'Pillow==10.0.1',
        'plotly==5.18.0',
        'pyaml==23.9.7',
        'pyparsing==3.1.1',
        'python-dateutil==2.8.2',
        'pytz==2023.3.post1',
        'PyYAML==6.0.1',
        'scikit-learn==1.3.1',
        'scikit-optimize==0.9.0',
        'scipy==1.11.3',
        'seaborn==0.13.0',
        'six==1.16.0',
        'tenacity==8.2.3',
        'threadpoolctl==3.2.0',
        'tzdata==2023.3',
        'xgboost==2.0.0' 
        ]
        
     ,
    author="Zalman Goldstein",
    author_email="zaalgol@gmail.com",
    description="A brief description of TabularWizard.",
    keywords="tabular data ML",
)
