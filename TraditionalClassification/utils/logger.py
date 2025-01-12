import os
import atexit
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Maximum log file size (10MB) and number of backup files
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB in bytes
MAX_BACKUP_COUNT = 50  # Number of backup files to keep


class RunLogger:
    _instance = None
    _run_timestamp = None
    _run_dir = None
    _loggers = {}
    _initialized = False

    def __init__(self):
        raise RuntimeError("Call initialize() instead")

    @classmethod
    def initialize(cls, base_log_dir="log/", suffix=""):
        """Initialize the RunLogger with a timestamp"""
        if not cls._initialized:
            cls._run_timestamp = datetime.now().strftime("%m%d_%H%M%S")
            cls._run_dir = os.path.join(
                base_log_dir, f"run_{cls._run_timestamp}_{suffix}"
            )
            os.makedirs(cls._run_dir, exist_ok=True)
            atexit.register(cls._cleanup_empty_run)
            cls._initialized = True

    @classmethod
    def get_run_directory(cls, base_log_dir="log/"):
        """Get the run-specific directory"""
        if not cls._initialized:
            cls.initialize(base_log_dir)
        return cls._run_dir

    @classmethod
    def _cleanup_empty_run(cls):
        """清理空的运行目录（如果程序启动后没有生成任何日志）"""
        if cls._run_dir and os.path.exists(cls._run_dir):
            if not os.listdir(cls._run_dir):  # 如果目录是空的
                try:
                    os.rmdir(cls._run_dir)
                except Exception:
                    pass


def setup_logger(name, log_file=None, log_dir="log/"):
    """Set up loggers with the given name, optionally logging to a file.
    Logger will output INFO level to console and DEBUG level to file if specified.

    Args:
        name: Logger name
        log_file: Optional log file path

    Returns:
        logging.Logger: Configured logger
    """
    if name in RunLogger._loggers:
        return RunLogger._loggers[name]

    logger = logging.getLogger(name)
    # Set to DEBUG to allow all log levels
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if log_file:
        run_dir = RunLogger.get_run_directory(log_dir)

        log_file = log_file.split("/")[-1]  # 只取最后一个文件名字
        log_file = log_file[:-4] if log_file.endswith(".log") else log_file
        log_file += ".log"

        log_file_path = os.path.join(run_dir, log_file)

        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=MAX_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


# def clean_old_run_directories(base_log_dir="log/", max_runs=60):
#     """Clean up old run directories keeping only the most recent ones"""
#     try:
#         # 确保基础目录存在
#         if not os.path.exists(base_log_dir):
#             return

#         run_dirs = [
#             d
#             for d in os.listdir(base_log_dir)
#             if d.startswith("run_") and os.path.isdir(os.path.join(base_log_dir, d))
#         ]
#         run_dirs = sorted(run_dirs, key=lambda x: x.split("_")[1], reverse=True)

#         # Remove old run directories
#         for old_dir in run_dirs[max_runs:]:
#             old_dir_path = os.path.join(base_log_dir, old_dir)
#             try:
#                 for file in os.listdir(old_dir_path):
#                     os.remove(os.path.join(old_dir_path, file))
#                 os.rmdir(old_dir_path)
#             except Exception as e:
#                 print(f"Error cleaning up directory {old_dir_path}: {e}")
#     except Exception as e:
#         print(f"Error during cleanup: {e}")


def clean_old_run_directories(base_log_dir="log/", max_runs=60):
    """Clean up old run directories keeping only the most recent ones.
    Rename directories with empty average_res.log by adding 'empty' suffix."""
    try:
        # Ensure base directory exists
        if not os.path.exists(base_log_dir):
            return

        run_dirs = [
            d
            for d in os.listdir(base_log_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(base_log_dir, d))
        ]

        # Check and rename directories with empty average_res.log
        for dir_name in run_dirs[:]:
            dir_path = os.path.join(base_log_dir, dir_name)
            log_file = os.path.join(dir_path, "average_res.log")

            try:
                if not os.path.exists(log_file) or (
                    os.path.getsize(log_file) == 0 and not dir_name.endswith("_empty")
                ):
                    # Rename the directory by adding '_empty' suffix
                    new_dir_name = f"{dir_name}_empty"
                    new_dir_path = os.path.join(base_log_dir, new_dir_name)
                    os.rename(dir_path, new_dir_path)
                    # Update the directory name in our list
                    run_dirs.remove(dir_name)
                    run_dirs.append(new_dir_name)
            except Exception as e:
                print(f"Error checking/renaming directory {dir_path}: {e}")

        # Sort remaining directories and remove old ones
        # Exclude '_empty' directories from max_runs count
        normal_dirs = [d for d in run_dirs if not d.endswith("_empty")]
        empty_dirs = [d for d in run_dirs if d.endswith("_empty")]

        normal_dirs = sorted(normal_dirs, key=lambda x: x.split("_")[1], reverse=True)

        # Remove old directories (only from non-empty ones)
        for old_dir in normal_dirs[max_runs:]:
            old_dir_path = os.path.join(base_log_dir, old_dir)
            try:
                for file in os.listdir(old_dir_path):
                    os.remove(os.path.join(old_dir_path, file))
                os.rmdir(old_dir_path)
            except Exception as e:
                print(f"Error cleaning up directory {old_dir_path}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


# 导出需要的函数和类
__all__ = ["setup_logger", "RunLogger", "clean_old_run_directories"]
