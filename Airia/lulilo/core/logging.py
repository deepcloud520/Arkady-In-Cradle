"""模块级 logger 工厂，替代裸 print()（AGENTS.md 4.2）。

用法:
    from Airia.lulilo.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("something happened")
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """创建或获取模块级 logger，带统一格式。

    Args:
        name: 通常传 ``__name__``。
        level: 默认 INFO；调试时可传 ``logging.DEBUG``。

    Returns:
        配置好 handler/formatter 的 Logger 实例。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-5s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
