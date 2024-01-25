import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "customVOC",
    version = "0.0.1",
    author = "Aljbri Abdussalam",
    author_email = "mr.aljbri@gmail.com",
    description = "Custom Pascal VOC Dataset reader for Pytorch",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/aljbri/customPascalVOC",
    project_urls = {
        "Bug Tracker": "https://github.com/aljbri/customPascalVOC/issues",
        "repository" : "https://github.com/aljbri/customPascalVOC"
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)
