from setuptools import setup, find_packages

setup(
    name='datawarden',
    version='0.0.1',
    description='This package is dedicated to providing cutting-edge tools and methodologies to evaluate and curate datasets specifically designed for Large Language Models (LLMs). Leveraging the capabilities of LLMs themselves, combined with programmatic best practices, our toolkit ensures a robust evaluation and refinement process for your datasets.',
    author='e-xperiments',
    author_email='ahm.rimer@gmail.com',
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        # List your dependencies here
        'transformers',
    ],
)
