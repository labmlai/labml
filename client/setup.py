import setuptools
import labml

with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml',
    version=labml.__version__,
    author="Varuna Jayasiri, Nipun Wijerathne",
    author_email="vpjayasiri@gmail.com, hnipun@gmail.com",
    description="Organize Machine Learning Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lab-ml/labml",
    project_urls={
        'Documentation': 'https://lab-ml.com/'
    },
    packages=setuptools.find_packages(exclude=('labml_helpers',
                                               'labml_helpers.*',
                                               'test',
                                               'test.*')),
    install_requires=['gitpython',
                      'pyyaml',
                      'numpy'],
    entry_points={
        'console_scripts': ['labml=labml.cli:main'],
    },
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
