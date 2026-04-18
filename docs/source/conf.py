import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "hmfast"
copyright = "2025, The hmfast developers"
author = "Patrick Janulewicz, Licong Xu, Boris Bolliet"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autosummary_generate = True
autosummary_imported_members = False

autodoc_member_order = "bysource"
autoclass_content = "class"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,  # Ensures private methods are not shown
    "special-members": "",     # Do not include __init__ or other dunder methods
    "show-inheritance": True,
    "inherited-members": True,
}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
