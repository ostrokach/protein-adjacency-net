"""Tests for `pagnn` package."""
import pytest

from click.testing import CliRunner

import pagnn
from pagnn import cli

@pytest.mark.parametrize('attribute', ['__author__', '__email__', '__version__'])
def test_attribute(attribute):
    assert getattr(pagnn, attribute)


def test_main():
    from pagnn import pagnn
    assert pagnn


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'pagnn.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
