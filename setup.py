import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

print(setuptools.find_packages())

setuptools.setup(
    name='lab',
    version='3.0',
    author="Varuna Jayasiri",
    author_email="vpjayasiri@gmail.com",
    description="🧪 Organize Machine Learning Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vpj/lab",
    packages=setuptools.find_packages(exclude=('samples', 'samples.*')),
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
