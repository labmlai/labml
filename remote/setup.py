import setuptools

with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml_remote',
    version='0.1.0',
    author="Varuna Jayasiri",
    author_email="vpjayasiri@gmail.com",
    description="Run python code on remote servers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lab-ml/remote",
    project_urls={
        'Documentation': 'https://lab-ml.com/'
    },
    packages=setuptools.find_packages(exclude=('test',
                                               'test.*')),
    include_package_data=True,
    install_requires=['paramiko',
                      'pyyaml>=5.3.1',
                      'scp',
                      'click'],
    entry_points={
        'console_scripts': ['labml_remote=labml_remote.cli:main'],
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
