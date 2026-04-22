import logging

import pytest

from config import setup_logging
from errors import ParseError
from kicad_interface.parser import parse_kicad_sch


def test_invalid_kicad_sch_raises_parse_error():
    with pytest.raises(ParseError):
        parse_kicad_sch("(not_a_kicad_file)")


def test_setup_logging_writes_run_log(tmp_path):
    log_file = tmp_path / "run.log"
    logger = setup_logging(log_file)
    logger.info("logging smoke test")

    for handler in logging.getLogger().handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()

    content = log_file.read_text(encoding="utf-8")
    assert "logging smoke test" in content
