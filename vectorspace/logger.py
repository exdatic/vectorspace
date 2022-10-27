from contextlib import contextmanager
import logging
from os import environ
import sys
import time
from typing import Any, Dict, Optional, Union

from loguru import logger

from vectorspace.utils import humantime

DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentaion.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def init_logger(
    filter: Dict[str, Union[bool, str]] = {},
    format: str = DEFAULT_FORMAT,
    level: str = 'INFO',
    diagnose: bool = False,
):
    """
    Replace standard logging with loguru
    """
    intercept_handler = InterceptHandler()

    # replace root handler by intercept handler
    logging.getLogger().handlers = [intercept_handler]

    # uvicorn       : propagate=True,  stream=sys.stderr
    # uvicorn.access: propagate=False, stream=sys.stdout

    for log in logging.root.manager.loggerDict.values():
        if hasattr(log, 'handlers'):
            assert isinstance(log, logging.Logger)
            log.handlers = []
            log.propagate = True

    # set logs output, level and format
    logger.configure(
        handlers=[{
            'sink': sys.stdout,
            'level': environ.get('LOGURU_LEVEL', level),
            'filter': filter,
            'format': format,
            'diagnose': diagnose,
            'enqueue': True,  # non-blocking
            'colorize': True
        }]
    )


@contextmanager
def log_time(msg: str, color: Optional[str] = None):
    """Log message before and after this block, including the duration and an optional result value"""
    class Result:
        def __init__(self):
            self.value = None

        def set(self, value: Any):
            self.value = value

    def info(msg: str):
        if color:
            logger.opt(colors=True).info(f"<{color}>{msg}</{color}>")
        else:
            logger.info(msg)

    info(msg)
    start_time = time.time()
    result = Result()
    try:
        yield result
    finally:
        end_time = time.time()
        duration = end_time - start_time
        if result.value is not None:
            info(f"{msg} took {humantime(duration)} ({result.value})")
        else:
            info(f"{msg} took {humantime(duration)}")
