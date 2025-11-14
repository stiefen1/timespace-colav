"""
Time-space collision avoidance library
"""

__version__ = "0.1.0"

# Import main modules
from .path import *
from .timespace import *
from .obstacles import *
from .utils import *
from .colregs import *
from .planner import *

# Library logging setup: expose a package logger but don't configure handlers by default.
# This prevents 'No handlers could be found' warnings while letting applications decide formatting.
import logging as _logging
_logger = _logging.getLogger(__name__)
_logger.addHandler(_logging.NullHandler())

def get_logger(name: str | None = None) -> _logging.Logger:
	"""Return a namespaced logger within the colav hierarchy.

	Example:
		logger = colav.get_logger("planner")
		logger.info("Planning started")
	"""
	base = __name__ if name is None else f"{__name__}.{name}"
	return _logging.getLogger(base)

def configure_logging(
	level: int = _logging.INFO,
	fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
	datefmt: str | None = "%H:%M:%S",
	colored: bool = True,
) -> _logging.Handler:
	"""Optionally configure console logging for the 'colav' package only.

	- Leaves the root logger unchanged (keeps other libs quiet unless the app configures them)
	- Adds a StreamHandler to the 'colav' logger
	- Filters so only 'colav' (and children) appear on this handler
	- Disables propagation to avoid duplicates
	- If ``colored`` is True, applies level-based colors (blue=DEBUG, green=INFO,
	  yellow=WARNING, red=ERROR/CRITICAL). On Windows, uses colorama when available.

	Returns the handler so callers can remove or tweak it later.
	"""
	import sys as _sys

	# Try to enable ANSI colors on Windows if requested
	if colored:
		try:
			import colorama as _colorama  # type: ignore
			# Prefer just_fix_windows_console for modern consoles
			if hasattr(_colorama, "just_fix_windows_console"):
				_colorama.just_fix_windows_console()
			else:
				_colorama.init()
		except Exception:
			# If colorama isn't available, we'll still proceed without it
			pass

	class _LevelColorFormatter(_logging.Formatter):
		_RESET = "\x1b[0m"
		_COLORS = {
			_logging.DEBUG:   "\x1b[34m",          # blue
			_logging.INFO:    "\x1b[32m",          # green
			_logging.WARNING: "\x1b[33m",          # yellow (orange-ish)
			_logging.ERROR:   "\x1b[31m",          # red
			_logging.CRITICAL:"\x1b[1m\x1b[31m",  # bold red
		}

		def format(self, record: _logging.LogRecord) -> str:
			if colored:
				color = self._COLORS.get(record.levelno, "")
				original = record.levelname
				record.levelname = f"{color}{original}{self._RESET}"
				try:
					return super().format(record)
				finally:
					record.levelname = original
			else:
				return super().format(record)

	pkg_logger = _logging.getLogger(__name__)
	pkg_logger.setLevel(level)

	handler = _logging.StreamHandler(_sys.stdout)
	handler.setLevel(level)
	handler.setFormatter(_LevelColorFormatter(fmt=fmt, datefmt=datefmt))
	handler.addFilter(lambda rec: rec.name == __name__ or rec.name.startswith(f"{__name__}."))

	# Avoid duplicate logs if root has handlers
	pkg_logger.propagate = False

	# Remove existing non-Null handlers to prevent duplicates on repeated calls
	for h in list(pkg_logger.handlers):
		if not isinstance(h, _logging.NullHandler):
			pkg_logger.removeHandler(h)

	pkg_logger.addHandler(handler)
	return handler
