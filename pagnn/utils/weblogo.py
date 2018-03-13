import io
import logging
from typing import List
from unittest.mock import patch

import weblogolib._cli
from PIL import Image

logger = logging.getLogger(__name__)


class _BytesIO(io.BytesIO):

    def __init__(self):
        super().__init__()
        self.buffer = self


def make_weblogo(seqs: List[str],
                 units: str = 'bits',
                 color_scheme: str = 'charge',
                 stacks_per_line: int = 60):
    weblogo_args = [
        'weblogo',
        f'--format=png',
        f'--units={units}',
        '--sequence-type=protein',
        f'--stacks-per-line={stacks_per_line}',
        f'--color-scheme={color_scheme}',
        '--scale-width=no',
    ]

    fin = io.StringIO()
    _write_sequences(seqs, fin)
    fin.seek(0)

    with patch('sys.stdin', fin), patch('weblogolib._cli.sys.argv', weblogo_args), \
            patch('sys.stdout', new_callable=_BytesIO) as patch_out:
        try:
            weblogolib._cli.main()
        except RuntimeError as e:
            logger.error("Failed to create WebLogo image because of error: '%s'.", str(e))
            img = None
        else:
            img = Image.open(patch_out)

    return img


def _write_sequences(seqs, fh):
    for i in range(len(seqs)):
        fh.write(f'> seq_{i}\n')
        fh.write(seqs[i] + '\n')
