# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TimeSeriesForecasting"
copyright = "2025, Engelsgeduld"
author = "Engelsgeduld"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))  # Ensure it finds your module

# Enable extensions
extensions = [
    "sphinx.ext.autodoc",  # Extracts docstrings
    "sphinx.ext.napoleon",  # Supports Google-style docstrings
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx.ext.todo",  # Supports TODOs
    "sphinx.ext.githubpages",  # Supports GitHub Pages hosting
]

# Use ReadTheDocs theme (SciPy style)
html_theme = "sphinx_rtd_theme"

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
