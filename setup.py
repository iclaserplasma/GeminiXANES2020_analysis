from setuptools import setup, find_packages
import os
import XANES2020_code

# currently looking 3 levels down, didn't seem to get the recursive code working....
data_files = ['pkg_data/*','pkg_data/*/*','pkg_data/*/*/*']
# for root, dirs, files in os.walk(os.path.join(__name__,'pkg_data')):
#    for name in dirs:
#       data_files.append(os.path.join(root, name,'*'))

# print(data_files)
setup(name='XANES2020_code',
      version='0.1.1',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      package_data = {'XANES2020_code':data_files},
      install_requires=['opencv-python','scipy','scikit-learn',
                        'scikit-image','scikit-optimize','Pillow','PyQt5',
                        'pandas','pyqtgraph'],
      )
