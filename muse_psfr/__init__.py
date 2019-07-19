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

from pkg_resources import get_distribution, DistributionNotFound  # noqa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass
