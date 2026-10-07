"""Sphinx configuration for the rapidtools documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _version

# Make the package importable when the docs are built from a plain checkout:
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'rapidtools'
copyright = '2025, University of Washington'
author = 'UW RAPID'
try:
    release = _version('rapidtools')
except PackageNotFoundError:  # pragma: no cover - building without an install
    release = '0.0.0'
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinx_copybutton',
]
templates_path = ['_templates']
exclude_patterns = []

# Autodoc / autosummary: one page per public class, short signatures.
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True
autodoc_default_options = {'show-inheritance': True}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None),
}

copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_book_theme'
html_title = 'rapidtools'
html_static_path = ['_static']
html_css_files = ['custom.css']
# Square icon cut from the wordmark's red "r" (ICO carries 16-64 px sizes):
html_favicon = '_static/favicon.ico'

# The primary sidebar: two linked logos replace the default title, followed by
# the theme's icon links, search box and navigation tree.
html_sidebars = {
    '**': [
        'rapid-logos.html',
        'icon-links.html',
        'sbt-sidebar-nav.html',
    ]
}

html_theme_options = {
    'repository_url': 'https://github.com/RAPID-Facility/rAPIdtools',
    'repository_branch': 'main',
    'path_to_docs': 'docs/source',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_download_button': False,
    'use_fullscreen_button': False,
    'home_page_in_toc': True,
    'show_navbar_depth': 1,
    'show_toc_level': 2,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/RAPID-Facility/rAPIdtools',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/rapidtools/',
            'icon': 'fa-brands fa-python',
        },
        {
            'name': 'UW RAPID Facility',
            'url': 'https://www.uwrapid.org/',
            'icon': 'fa-solid fa-house-flood-water',
        },
        {
            'name': 'Feedback and feature requests',
            'url': 'https://github.com/RAPID-Facility/rAPIdtools/issues/new/choose',
            'icon': 'fa-regular fa-comment-dots',
        },
    ],
}
