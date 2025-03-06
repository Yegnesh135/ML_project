from setuptools import find_packages, setup
from typing import List


 #Though this is important in making the setup.py get executed after requirements.txt, it is not needed in install_requires.

# Define a function to make the install_reuires track the requirements.txt file automatically.
def get_requirements(file_path:str)->List[str]:
    '''
    This function return list of requirements
    '''
    Hypen_e_dot = '-e . '
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)
    # print(requirements)
    return requirements


setup(
    name='First_ML_End_to_End',
    version='0.0.1',
    author_email='g.yegnesh135@gmail.com',
    author='Yegnesh',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)   