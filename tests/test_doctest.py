import doctest
import logging
import os
import os.path as op
import tempfile

import numpy as np
import pytest

import pagnn
from kmtools import py_tools

logger = logging.getLogger(__name__)

DOCTEST_OPTIONFLAGS = (doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS |
                       doctest.IGNORE_EXCEPTION_DETAIL)

DOCTEST_EXTRAGLOBS = {'os': os, 'op': op, 'tempfile': tempfile, 'np': np}


@pytest.mark.parametrize("module_name, module", py_tools.iter_submodules(pagnn))
def test_doctest(module_name, module):
    failure_count, test_count = doctest.testmod(
        module, optionflags=DOCTEST_OPTIONFLAGS, extraglobs=DOCTEST_EXTRAGLOBS)
    assert failure_count == 0
