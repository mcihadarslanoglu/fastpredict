"""The setup script."""

from setuptools import setup, find_packages
from fastpredict import __version__,__author__, __email__

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md") as changelog_file:
    changelog = changelog_file.read()

requirements = [requirement for requirement in open('requirements.txt')]



setup(
    author=__author__,
    author_email=__email__,
    python_requires=">=3.6",
    description="Fastpredict is an improved alternative to Lazypredict project",
    long_description_content_type="text/markdown",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + changelog,
    include_package_data=True,
    keywords="fastpredict",
    name="fastpredict",
    packages=find_packages(include=["fastpredict", "fastpredict.*"]),
    url="https://github.com/mcihadarslanoglu/fastpredict",
    version=__version__,
    zip_safe=False,
)
