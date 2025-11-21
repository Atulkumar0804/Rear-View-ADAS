"""
logger.py
---------
Lightweight logging utility with configurable log levels.

Usage:
    from utils.logger import Logger
    log = Logger("app.log")
    log.info("Started system")
    log.warn("Low FPS")
    log.error("Critical issue")
"""

import os
import time


class Logger:
    LEVELS = {
        "INFO": 1,
        "WARN": 2,
        "ERROR": 3,
    }

    def __init__(self, logfile="system.log", level="INFO"):
        self.logfile = logfile
        self.level = level.upper()
        os.makedirs(os.path.dirname(logfile), exist_ok=True) if "/" in logfile else None

    def _write(self, tag, msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{timestamp}] [{tag}] {msg}"
        print(text)
        with open(self.logfile, "a") as f:
            f.write(text + "\n")

    def info(self, msg):
        if self.LEVELS[self.level] <= 1:
            self._write("INFO", msg)

    def warn(self, msg):
        if self.LEVELS[self.level] <= 2:
            self._write("WARN", msg)

    def error(self, msg):
        if self.LEVELS[self.level] <= 3:
            self._write("ERROR", msg)
