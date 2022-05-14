def _setup_logging():
    from mpdaf.log import setup_logging
    import sys

    setup_logging(
        "muse_psfr",
        fmt="[%(levelname)s] %(message)s",
        level="INFO",
        color=True,
        stream=sys.stdout,
    )


_setup_logging()

from .psfrec import *  # noqa
from .version import version as __version__  # noqa
