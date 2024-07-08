import setuptools

with open("../readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='labml_app',
    version='0.0.103',
    author="Varuna Jayasiri, Nipun, Aditya",
    author_email="vpjayasiri@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/labmlai/labml",
    project_urls={
        'Documentation': 'https://docs.labml.ai'
    },
    install_requires=['labml>=0.4.158',
                      'gunicorn',
                      'numpy',
                      'labml-db',
                      'fastapi',
                      'uvicorn',
                      'aiofiles',
                      'labml_db',
                      'pymongo',
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
