#!/usr/bin/env python

"""Setup script."""

import os
import glob
import numpy
import yaml
from setuptools import setup, find_packages, Extension

# Package name
name = 'sugar'

# Packages (subdirectories in clusters/)
packages = find_packages()

# Scripts (in scripts/)
scripts = glob.glob("scripts/*.py")

package_data = {}

setup(name=name,
      description=("sugar"),
      classifiers=["Topic :: Scientific :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PFLeget",
      packages=packages,
      scripts=scripts)
