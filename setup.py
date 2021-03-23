from setuptools import setup, find_packages


def read_file(file):
    with open(file) as fin:
        return fin.read()


setup(
    name="pagnn",
    version="0.1.14",
    description="Protein Adjacency Graph Neural Network.",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="Alexey Strokach",
    author_email="alex.strokach@utoronto.ca",
    url="https://gitlab.com/ostrokach/protein-adjacency-net",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"pagnn": ["model_data/*.state", "training/data/*"]},
    include_package_data=True,
    zip_safe=False,
    keywords="pagnn",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
)
