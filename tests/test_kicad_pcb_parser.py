import pytest

from errors import ParseError
from kicad_interface.pcb_parser import parse_kicad_pcb


def test_header_only_kicad_pcb_is_rejected():
    content = '(kicad_pcb (version 20240108) (generator "pcbnew") (generator_version "8.0"))'

    with pytest.raises(ParseError, match="empty or unsupported|no footprints"):
        parse_kicad_pcb(content)


def test_non_kicad_input_is_rejected():
    with pytest.raises(ParseError, match="Not a valid KiCad PCB file"):
        parse_kicad_pcb('(not_a_board)')
