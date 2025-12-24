from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic_mapping'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), 
         glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'rviz'), 
         glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nguyen Anh Duc',
    maintainer_email='duc281004@gmail.com',
    description='Semantic Mapping with TurtleBot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'frontier_explorer_v2 = semantic_mapping.frontier_explorer_v2:main',
            'frontier_explorer = semantic_mapping.frontier_explorer:main',
            'semantic_mapper = semantic_mapping.semantic_mapper:main',
            'sim_detector = semantic_mapping.sim_detector_node:main',
            'oakd_detector = semantic_mapping.oakd_detector_node:main',
        ],
    },
)