import setuptools

with open("../readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml_app',
    version='0.5.12',
    author="labml.ai Team",
    author_email="contact@labml.ai",
    description="Web app for https://github.com/labmlai/labml",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/labmlai/labml",
    project_urls={
        'Documentation': 'https://docs.labml.ai'
    },
    install_requires=['labml>=0.5.2',
                      'gunicorn>=22.0.0',
                      'numpy',
                      'labml-db>=0.0.15',
                      'fastapi>=0.111.0',
                      'uvicorn>=0.30.1',
                      'pymongo>=4.8.0',
                      ],
    packages=['labml_app'],
    include_package_data=True,
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
