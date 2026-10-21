"""Deprecated import alias for :mod:`ccutils`.

Use ``import ccutils`` in new code. The top-level name ``parser`` may resolve to
Python's standard-library module on older Python versions.
"""

from ccutils import *  # noqa: F401,F403
from ccutils import __all__
