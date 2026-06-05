"""
Logging setup utilities for TorchDock pipeline.

Provides a colored console formatter, file logging support, and a
convenience function for creating named loggers with consistent formatting.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import os
import logging

# Global log dictionary, for storing different named loggers
_loggers = {}

# Log level mapping
LOG_LEVEL_MAP = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


class LogFormatter(logging.Formatter):
    """Base log formatter with optional color support."""

    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[1;91m',
        'RESET': '\033[0m'
    }

    def __init__(self, use_color=False):
        super().__init__()
        self.use_color = use_color

    def format(self, record):
        # Add call stack info for ERROR and above
        if record.levelno >= logging.ERROR:
            record.callinfo = f"{record.filename}:{record.funcName}:{record.lineno}"
            log_fmt = "[%(levelname)s] %(asctime)s - %(callinfo)s - %(message)s"
        else:
            log_fmt = "[%(levelname)s] %(asctime)s - %(message)s"

        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        formatted_message = formatter.format(record)

        # Add color for console output
        if self.use_color and record.levelname in self.COLORS:
            formatted_message = self.COLORS[record.levelname] + formatted_message + self.COLORS['RESET']

        return formatted_message


def setup_logger(name='torchdock', level='info', log_file=None, console_output=True):
    """Set up and return a named logger.

    Args:
        name (str): Logger name.
        level (str): Log level, one of 'debug', 'info', 'warning', 'error', 'critical'.
        log_file (str, optional): Log file path. If None, no file output.
        console_output (bool): Whether to output to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL_MAP.get(level.lower(), logging.INFO))

    # Clear existing handlers to allow reconfiguration
    logger.handlers.clear()

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(LogFormatter(use_color=True))
        logger.addHandler(console_handler)

    # Add file handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(LogFormatter(use_color=False))
        logger.addHandler(file_handler)

    # Cache and return
    _loggers[name] = logger
    return logger
