import setuptools

with open("readme.rst", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='machine_learning_lab',
    version='0.4.0',
    author="Varuna Jayasiri, Nipun Wijerathne",
    author_email="vpjayasiri@gmail.com",
    description="🧪 Organize Machine Learning Experiments",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/lab-ml/lab",
    packages=setuptools.find_packages(exclude=('test',
                                               'test.*')),
    install_requires=['gitpython',
                      'pyyaml',
                      'numpy'],
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
