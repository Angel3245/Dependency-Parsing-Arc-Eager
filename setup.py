from setuptools import setup

setup(
   name='Dependency parsing',
   version='1.0',
   description='Implement a dependency parser using the arc-eager oracle',
   author='Jose Ángel Pérez Garrido',
   author_email='jpgarrido19@esei.uvigo.es',
   packages=['dp'],
   install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)