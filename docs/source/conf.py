import os
import sys
# Make sure to use an absolute path
sys.path.insert(0, os.path.abspath('../../rapidtools'))

project = 'rapidtools'
copyright = '2025, University of Washington'
author = 'UW RAPID'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Core Sphinx extension for auto-doc generation
    'sphinx.ext.napoleon',     # To understand Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # To add links to highlighted source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
