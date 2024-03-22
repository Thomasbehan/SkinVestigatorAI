import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.txt')) as f:
    CHANGES = f.read()

requires = [
    'plaster_pastedeploy',
    'pyramid',
    'pyramid_jinja2',
    'pyramid_debugtoolbar',
    'waitress',
    'Pillow==10.2.0',
    'requests==2.31.0',
    'scipy==1.10.1',
    'sympy==1.12',
    'tensorboard==2.12.2',
    'termcolor==2.3.0',
    'tensorflow==2.12.0',
    'torch==2.0.0',
    'torchvision==0.15.1',
    'tqdm==4.65.0',
    'vit-keras==0.1.2',
    'tensorflow_addons==0.20.0',
    'albumentations',
    'keras-tuner',
]

tests_require = [
    'WebTest',
    'pytest',
    'pytest-cov',
    'flake8',
]

setup(
    name='skinvestigatorai',
    version='2.0.0',
    description='SkinVestigatorAI',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='',
    author_email='',
    url='',
    keywords='web pyramid pylons',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'testing': tests_require,
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = skinvestigatorai:main',
        ],
    },
)
