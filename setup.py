
from distutils.core import setup

setup(
  name = 'CORG',         
  version = '0.2',      
  license='MIT',        
  description = 'CORG: Corporal geometrics for text dimensionality analysis',   
  author = 'Pedro Ramaciotti Morales',                      
  url = 'https://github.com/pedroramaciotti/CORG',   
  keywords = ['NLP'],
  install_requires=[            
          'numpy',
          'pandas',
          'scipy',
      ],
  packages = ["corg"],
  include_package_data=True,
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)