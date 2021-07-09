import setuptools

with open("readme.rst", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml-helpers',
    version='0.4.77',
    author="Varuna Jayasiri, Nipun Wijerathne",
    author_email="vpjayasiri@gmail.com, hnipun@gmail.com",
    description="A collection of classes and functions to automate common deep learning training patterns",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/lab-ml/helpers",
    project_urls={
        'Documentation': 'https://lab-ml.com/'
    },
    packages=setuptools.find_packages(exclude=('test',
                                               'test.*')),
    install_requires=['labml>=0.4.97',
                      'torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine learning',
)
