"""
Timespace collision avoidance library for maritime navigation.

Provides tools for maritime collision avoidance planning using time-space
projection and visibility graphs. Supports COLREGS compliance, moving
obstacle avoidance, and optimal path planning for autonomous vessels.

Main Components
---------------
TimeSpaceColav : High-level collision avoidance planner
MovingShip : Moving obstacle representation
TimeSpaceProjector : Projects obstacles into time-space
VGPathPlanner : Visibility graph path planning
COLREGS : Maritime collision regulation filters

Examples
--------
Basic collision avoidance:

>>> import colav
>>> planner = colav.TimeSpaceColav(desired_speed=10.0, colregs=True)
>>> ship = colav.MovingShip((100, 0), 90, (5, 0), 8, 3, mmsi=123)
>>> trajectory, info = planner.get((0, 0), (200, 100), [ship])
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
	"""
	Return a namespaced logger within the colav hierarchy.
	
	Parameters
	----------
	name : str, optional
		Logger name suffix. If None, returns the base colav logger.
		
	Returns
	-------
	Logger
		Configured logger instance for the colav package.
		
	Examples
	--------
	>>> logger = colav.get_logger("planner")
	>>> logger.info("Planning started")
	"""
	base = __name__ if name is None else f"{__name__}.{name}"
	return _logging.getLogger(base)

def configure_logging(
	level: int = _logging.INFO,
	fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
	datefmt: str | None = "%H:%M:%S",
	colored: bool = True,
) -> _logging.Handler:
	"""
	Configure console logging for the colav package.
	
	Sets up colored console output with filtering to show only colav
	package messages. Leaves other loggers unchanged.
	
	Parameters
	----------
	level : int, default INFO
		Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
	fmt : str, default includes timestamp and level
		Log message format string.
	datefmt : str, optional
		Date format for timestamps. Defaults to "HH:MM:SS".
	colored : bool, default True
		Enable color-coded log levels in console output.
		
	Returns
	-------
	Handler
		The created stream handler for potential removal or modification.
		
	Examples
	--------
	>>> import colav
	>>> colav.configure_logging(level=colav.logging.DEBUG)
	>>> logger = colav.get_logger("planner")
	>>> logger.debug("Debug message")  # Now visible
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
