"""
Sphinx Configuration File
=========================

For more info, see: http://www.sphinx-doc.org/en/stable/config.html
"""
import sys
import os

import sinusoidal_sphinx_theme
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify


def setup(app):
    app.add_stylesheet('css/custom.css')
    app.add_config_value('recommonmark_config', {
        'enable_eval_rst': True,
    }, True)
    app.add_transform(AutoStructify)


project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

extensions = [
    'sphinx.ext.mathjax',
    'sinusoidal_sphinx_theme',
]

templates_path = ['_templates']

source_suffix = ['.rst', '.md']

source_parsers = {
    '.md': CommonMarkParser,
}

master_doc = 'index'

project = u'Adjacency Net'

copyright = u"2018, Alexey Strokach"

version = '0.1.0'

release = '0.1.0'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

pygments_style = 'sphinx'

html_theme = 'sinusoidal_sphinx_theme'

html_theme_options = {
    "project_nav_name":
    project,
    "github_link":
    "https://gitlab.com/biological-data-warehouse/adjacency-net/",
}

html_theme_path = sinusoidal_sphinx_theme.html_theme_path()

html_static_path = ['_static']

htmlhelp_basename = 'adjacency-netdoc'

latex_elements: dict = {}

latex_documents = [
    ('index', 'adjacency-net.tex',
     u'Adjacency Net Documentation', u'Alexey Strokach', 'manual'),
]

man_pages = [('index', 'adjacency-net',
              u'Adjacency Net Documentation', [u'Alexey Strokach'],
              1)]

texinfo_documents = [
    ('index', 'adjacency-net', u'Adjacency Net Documentation',
     u'Alexey Strokach', 'adjacency-net',
     'One line description of project.', 'Miscellaneous'),
]
