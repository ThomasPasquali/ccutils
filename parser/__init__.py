"""CCUTILS output parsing helpers.

For installed use, the same API is available as :mod:`ccutils_parser`.
"""

from .ccutils_parser import (
    Document,
    MPIOutputParser,
    MPIAllPrint,
    ParserError,
    ParserPatterns,
    Section,
    TimerRecord,
    describe_structure,
    pretty_print_structure,
    parse,
    parse_ccutils,
    parse_ccutils_file,
    parse_ccutils_output,
    parse_ccutils_output_file,
    parse_file,
    parse_timer_records,
)

__all__ = [
    "Document", "MPIOutputParser", "MPIAllPrint", "ParserError",
    "ParserPatterns", "Section", "TimerRecord", "parse", "parse_ccutils",
    "describe_structure", "pretty_print_structure", "parse_ccutils_file",
    "parse_ccutils_output", "parse_ccutils_output_file",
    "parse_file", "parse_timer_records",
]
