from setuptools import setup, find_packages


def read_md(file):
    with open(file) as fin:
        return fin.read()


setup(
    name="pagnn",
    version="0.1.10",
    description="Protein Adjacency Graph Neural Network.",
    long_description=read_md("README.md"),
    author="Alexey Strokach",
    author_email="alex.strokach@utoronto.ca",
    url="https://gitlab.com/kimlab/pagnn",
    packages=find_packages(),
    package_data={"pagnn.prediction": "data/*", "pagnn.training": "data/*"},
    include_package_data=True,
    zip_safe=False,
    keywords="pagnn",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite="tests",
)
