# -- Project information -----------------------------------------------------
project = 'scaLR'
copyright = '2024, Infocusp Innovations'
author = 'Infocusp Innovations'
release = 'v1.0.0'

# -- General configuration ---------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.extlinks',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    # 'nbsphinx'
]
source_suffix = ['.rst', '.md']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = 'index'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['logo.css']
html_logo = "../img/scaLR_logo.png"
html_favicon = "../img/scaLR_logo.png"
