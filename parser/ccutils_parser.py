from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

ANSI = re.compile(r'(?:\x1b)?\[\d{1,3}(?:;\d{1,3})*m')
NEVER = r'(?!x)x'   # a pattern that never matches (disables a marker type)
TIMER_PREFIX = 'CCUTILS_TIMER '
TIMER_FIELD = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?:"(?P<quoted>(?:\\.|[^"\\])*)"|(?P<bare>[^\s]+))'
)


class ParserError(Exception):
    """Base exception for parser errors."""


class SectionFormatError(ParserError):
    """Invalid section structure."""


class MPIAllPrintFormatError(ParserError):
    """Invalid MPI ALL PRINT / rank structure."""


@dataclass
class TimerRecord:
    """One CCUTILS_TIMER v=1 record; timer-specific values are in `fields`."""
    version: int
    record_type: str
    fields: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    raw_line: str = ''

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def get(self, key: str, default=None) -> Any:
        return self.fields.get(key, default)

    def __getattr__(self, key: str) -> Any:
        # Make common fields natural to access (record.name, record.avg_ms).
        values = self.__dict__.get('fields', {})
        if key in values:
            return values[key]
        raise AttributeError(key)


def _parse_timer_line(line: str, line_number: int = 0) -> TimerRecord:
    parsed_line = ANSI.sub('', line)
    if not parsed_line.startswith(TIMER_PREFIX):
        raise ParserError('Not a CCUTILS_TIMER record')
    payload = parsed_line[len(TIMER_PREFIX):]
    values: Dict[str, Any] = {}
    pos = 0
    while pos < len(payload):
        while pos < len(payload) and payload[pos].isspace():
            pos += 1
        if pos == len(payload):
            break
        match = TIMER_FIELD.match(payload, pos)
        if match is None:
            raise ParserError(f'Malformed timer record at line {line_number}')
        key = match.group('key')
        if key in values:
            raise ParserError(f"Duplicate timer field '{key}' at line {line_number}")
        if match.group('quoted') is not None:
            try:
                value = json.loads('"' + match.group('quoted') + '"')
            except json.JSONDecodeError as error:
                raise ParserError(f'Invalid quoted timer field at line {line_number}: {error}') from error
        else:
            bare = match.group('bare')
            try:
                value = int(bare)
            except ValueError:
                try:
                    value = float(bare)
                except ValueError:
                    value = bare
        values[key] = value
        pos = match.end()

    version, record_type = values.pop('v', None), values.pop('type', None)
    if not isinstance(version, int) or not isinstance(record_type, str):
        raise ParserError(f'Timer record at line {line_number} requires integer v and string type')
    if version != 1:
        raise ParserError(f'Unsupported timer record version {version} at line {line_number}')
    return TimerRecord(version, record_type, values, line_number, line)


def parse_timer_records(text: str, strict: bool = False) -> List[TimerRecord]:
    """Return timer records from mixed CCUTILS/application output.

    Unrelated lines are ignored. Malformed timer lines are skipped unless
    `strict=True`, in which case a ParserError is raised.
    """
    result = []
    for line_number, line in enumerate(str(text).splitlines(), 1):
        if not ANSI.sub('', line).startswith(TIMER_PREFIX):
            continue
        try:
            result.append(_parse_timer_line(line, line_number))
        except ParserError:
            if strict:
                raise
    return result


@dataclass
class ParserPatterns:
    """Marker regexes, matched against one (ANSI-stripped) line each. Edit to fit other formats.
    Required named groups: name (section/json/mpi markers), title (optional), rank (rank markers)."""
    section_start: str = r'^\s*(?:=\+){2,}=?\s+(?P<name>\S+)\s+::\s+(?P<title>.+?)\s+(?:=\+){2,}=?\s*$'
    section_end: str = r'^\s*(?:=\+){2,}=?\s+(?P<name>\S+)\s+END\s+(?:=\+){2,}=?\s*$'
    global_json_start: str = r'^\s*(?:=-){2,}=?\s+(?P<name>\S+)\s+(?:=-){2,}=?\s*$'
    global_json_end: str = r'^\s*(?:=-){2,}=?\s+(?P<name>\S+)\s+END\s+(?:=-){2,}=?\s*$'
    mpi_start: str = r'^\s*(?:-\+){2,}-?\s+(?P<name>.+?)\s+(?:-\+){2,}-?\s*$'
    mpi_end: str = r'^\s*(?:-\+){2,}-?\s+(?P<name>.+?)\s+END\s+(?:-\+){2,}-?\s*$'
    rank_start: str = r'^\s*\[\[\s*Rank\s+(?P<rank>\d+)\s*\]\]\s*$'
    rank_end: str = r'^\s*\[\[\s*END\s+Rank\s+(?P<rank>\d+)\s*\]\]\s*$'

    def __post_init__(self):
        self._c = {f.name: re.compile(getattr(self, f.name)) for f in fields(self)}

    def match(self, kind: str, text: str):
        return self._c[kind].match(text)

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'ParserPatterns':
        return cls(**d)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> 'ParserPatterns':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def lenient(cls) -> 'ParserPatterns':
        """Any ===/---/###/+++ banner is a (nestable) section; `name :: title` is optional."""
        b = r'[=+\-#*~_]{3,}'
        return cls(
            section_start=rf'^\s*{b}\s+(?P<name>[^:]+?)(?:\s*::\s*(?P<title>.+?))?\s+{b}\s*$',
            section_end=rf'^\s*{b}\s+(?P<name>[^:]+?)\s+END\s+{b}\s*$',
            global_json_start=NEVER, global_json_end=NEVER, mpi_start=NEVER, mpi_end=NEVER)


@dataclass
class Line:
    no: int                       # 1-based line number
    text: str                     # ANSI-stripped text
    section: str = ''             # "outer/inner" path of the enclosing section
    mpi: Optional[str] = None     # enclosing MPI ALL PRINT name
    rank: Optional[int] = None    # enclosing rank
    kind: str = 'text'            # text | marker | json | json_body | timer
    data: Any = None              # parsed JSON (one-line JSON, or on the start marker of a JSON block)


@dataclass
class RankOutput:
    rank: int
    start_line: int = 0
    end_line: int = 0
    complete: bool = False
    lines: List[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return '\n'.join(l.text for l in self.lines)

    def json(self) -> List[Any]:
        """All JSON values found on single lines of this rank's output."""
        return [l.data for l in self.lines if l.kind == 'json']


@dataclass
class MPIAllPrint:
    name: str
    start_line: int = 0
    end_line: int = 0
    complete: bool = False
    ranks: Dict[int, RankOutput] = field(default_factory=dict)
    lines: List[Line] = field(default_factory=list)   # everything between the markers

    @property
    def raw_text(self) -> str:
        return '\n'.join(l.text for l in self.lines)

    @property
    def rank_outputs(self) -> Dict[int, str]:          # backwards compatible view
        return {r: o.text for r, o in self.ranks.items()}

    def get_rank_output(self, rank: int) -> Optional[str]:
        return self.ranks[rank].text if rank in self.ranks else None

    def get_all_ranks(self) -> List[int]:
        return sorted(self.ranks)

    def per_rank_json(self) -> Dict[int, Any]:
        """First JSON value per rank (typical for 'print JSON on every rank')."""
        return {r: o.json()[0] for r, o in sorted(self.ranks.items()) if o.json()}

    def per_rank_json_values(self) -> Dict[int, List[Any]]:
        """All JSON values emitted by each rank in this MPI print block."""
        return {r: values for r, o in sorted(self.ranks.items())
                if (values := o.json())}

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'lines': [self.start_line, self.end_line],
            'complete': self.complete,
            'ranks': {r: {'text': o.text, 'json': o.json(),
                          'lines': [o.start_line, o.end_line],
                          'complete': o.complete}
                      for r, o in sorted(self.ranks.items())},
            'text': self.raw_text,
        }


@dataclass
class Section:
    name: str
    title: str = ''
    path: str = ''
    start_line: int = 0
    end_line: int = 0
    complete: bool = False
    lines: List[Line] = field(default_factory=list)   # own lines (no MPI / JSON blocks, no children)
    json_blocks: Dict[str, Any] = field(default_factory=dict)
    mpi_all_prints: Dict[str, MPIAllPrint] = field(default_factory=dict)
    children: List['Section'] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        return '\n'.join(l.text for l in self.lines).strip()

    @property
    def json_data(self) -> Optional[Any]:
        return next(iter(self.json_blocks.values()), None)

    def get_global_json(self) -> Optional[Any]:
        return self.json_data

    def get_mpi_print(self, name: str) -> Optional[MPIAllPrint]:
        return self.mpi_all_prints.get(name)

    def get_child(self, name: str) -> Optional['Section']:
        """Return a direct subsection by name, or None if it is absent."""
        return next((child for child in self.children if child.name == name), None)

    def list_mpi_prints(self) -> List[str]:
        return list(self.mpi_all_prints)

    def walk(self) -> Iterator['Section']:
        yield self
        for c in self.children:
            yield from c.walk()

    def to_dict(self) -> dict:
        return {'name': self.name, 'title': self.title, 'lines': [self.start_line, self.end_line],
                'complete': self.complete, 'text': self.raw_text, 'json': self.json_blocks,
                'mpi': {n: m.to_dict() for n, m in self.mpi_all_prints.items()},
                'children': [c.to_dict() for c in self.children]}


@dataclass
class Document:
    lines: List[Line] = field(default_factory=list)            # every input line, annotated
    sections: Dict[str, Section] = field(default_factory=dict)  # top-level sections
    outside: List[Line] = field(default_factory=list)          # text outside any section
    mpi_all_prints: Dict[str, MPIAllPrint] = field(default_factory=dict)  # MPI blocks outside sections
    json_blocks: Dict[str, Any] = field(default_factory=dict)
    timer_records: List[TimerRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    marker_count: int = 0

    def __getitem__(self, name: str) -> Section:
        return self.sections[name]

    def __contains__(self, name: str) -> bool:
        return name in self.sections

    def __iter__(self):
        return iter(self.sections)

    def get(self, name: str, default=None):
        return self.sections.get(name, default)

    @property
    def json_data(self) -> Optional[Any]:
        """First top-level JSON block, if any (section blocks live on Section)."""
        return next(iter(self.json_blocks.values()), None)

    def get_mpi_print(self, name: str) -> Optional[MPIAllPrint]:
        """Get a top-level MPI print block by name."""
        return self.mpi_all_prints.get(name)

    def list_mpi_prints(self) -> List[str]:
        return list(self.mpi_all_prints)

    def all_sections(self) -> List[Section]:
        return [s for top in self.sections.values() for s in top.walk()]

    def find(self, name: str) -> List[Section]:
        """All sections called `name`, at any depth (handles duplicates)."""
        return [s for s in self.all_sections() if s.name == name]

    def to_dict(self) -> dict:
        return {'sections': {k: s.to_dict() for k, s in self.sections.items()},
                'outside': '\n'.join(l.text for l in self.outside),
                'json': self.json_blocks,
                'mpi': {n: m.to_dict() for n, m in self.mpi_all_prints.items()},
                'timer_records': [{'version': r.version, 'type': r.record_type,
                                   'fields': r.fields, 'line': r.line}
                                  for r in self.timer_records],
                'warnings': self.warnings}

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    def summary(self) -> str:
        timer_label = 'timer record' if len(self.timer_records) == 1 else 'timer records'
        warning_label = 'warning' if len(self.warnings) == 1 else 'warnings'
        out = [f'{len(self.lines)} lines, {self.marker_count} markers, '
               f'{len(self.timer_records)} {timer_label}, '
               f'{len(self.warnings)} {warning_label}']

        def show(s: Section, d: int):
            pad = '  ' * d
            out.append(f'{pad}{s.name}' + (f' :: {s.title}' if s.title else '') +
                       f'  [lines {s.start_line}-{s.end_line}]' + ('' if s.complete else '  (open)'))
            for n, m in s.mpi_all_prints.items():
                out.append(f'{pad}  mpi {n}: {len(m.ranks)} ranks')
            out.extend(f'{pad}  json {k}' for k in s.json_blocks)
            for c in s.children:
                show(c, d + 1)
        for s in self.sections.values():
            show(s, 0)
        out.extend(f'json {name}' for name in self.json_blocks)
        out.extend(f'mpi {name}: {len(block.ranks)} ranks'
                   for name, block in self.mpi_all_prints.items())
        return '\n'.join(out + [f'! {w}' for w in self.warnings])


def _unique(d: dict, name: str) -> str:
    if name not in d:
        return name
    n = 2
    while f'{name}#{n}' in d:
        n += 1
    return f'{name}#{n}'


class MPIOutputParser:
    """
    Args:
        patterns: ParserPatterns (default: CCUTILS format).
        strict: raise on structural problems instead of recording `doc.warnings`.
        strip_ansi: remove colour codes (also when the ESC byte was lost, e.g. "[96m").
        parse_json_lines: parse lines that look like one JSON value into `Line.data`.
    """

    def __init__(self, patterns: Optional[ParserPatterns] = None, strict: bool = False,
                 strip_ansi: bool = True, parse_json_lines: bool = True):
        self.patterns = patterns or ParserPatterns()
        self.strict, self.strip_ansi, self.parse_json_lines = strict, strip_ansi, parse_json_lines

    # -- backwards-compatible API (returns the top-level sections dict) --
    def parse_file(self, filepath: Union[str, Path]) -> Dict[str, Section]:
        return self.parse_document_file(filepath).sections

    def parse_string(self, content: str) -> Dict[str, Section]:
        return self.parse_document(content).sections

    def parse_document_file(self, filepath: Union[str, Path]) -> Document:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f'File not found: {p}')
        return self.parse_document(p.read_text(encoding='utf-8', errors='replace'))

    def parse_document(self, content: str) -> Document:
        P, doc = self.patterns, Document()
        stack: List[Section] = []
        mpi: Optional[MPIAllPrint] = None
        rank: Optional[RankOutput] = None
        jb: Optional[list] = None        # [name, start Line, body lines, owner Section|None]
        last = 0

        def fail(exc, msg):
            if self.strict:
                raise exc(msg)
            doc.warnings.append(msg)

        def close_rank(no, ok):
            nonlocal rank
            if rank:
                rank.end_line, rank.complete = no, ok
                if not ok:
                    fail(MPIAllPrintFormatError, f'Rank {rank.rank} (line {rank.start_line}) is not closed')
                rank = None

        def close_mpi(no, ok):
            nonlocal mpi
            close_rank(no, False)
            if mpi:
                mpi.end_line, mpi.complete = no, ok
                if not ok:
                    fail(MPIAllPrintFormatError, f"MPI print '{mpi.name}' (line {mpi.start_line}) is not closed")
                mpi = None

        def finish_json():
            nonlocal jb
            name, start, body, owner = jb
            try:
                data = json.loads('\n'.join(body).strip() or 'null', parse_constant=lambda c: None)
            except ValueError as e:
                data = None
                fail(ParserError, f"Invalid JSON block '{name}' at line {start.no}: {e}")
            start.data = data
            box = owner.json_blocks if owner else doc.json_blocks
            box[_unique(box, name)] = data
            jb = None

        for no, raw in enumerate(content.splitlines(), 1):
            last = no
            text = ANSI.sub('', raw) if self.strip_ansi else raw
            cur = stack[-1] if stack else None
            ln = Line(no, text, '/'.join(s.name for s in stack), mpi.name if mpi else None,
                      rank.rank if rank else None)
            doc.lines.append(ln)

            if jb:                                           # inside a global JSON block
                m = P.match('global_json_end', text)
                if m and m['name'] == jb[0]:
                    ln.kind = 'marker'
                    finish_json()
                else:
                    ln.kind = 'json_body'
                    jb[2].append(text)
                continue

            if text.startswith(TIMER_PREFIX):
                try:
                    timer_record = _parse_timer_line(text, no)
                except ParserError as error:
                    fail(ParserError, str(error))
                else:
                    ln.kind, ln.data = 'timer', timer_record
                    doc.timer_records.append(timer_record)
                (rank.lines if rank else mpi.lines if mpi else cur.lines if cur else doc.outside).append(ln)
                if rank and mpi:
                    mpi.lines.append(ln)
                continue

            if m := P.match('section_end', text):
                ln.kind, name = 'marker', m['name']
                doc.marker_count += 1
                if any(s.name == name for s in stack):
                    close_mpi(no - 1, False)
                    while True:
                        s = stack.pop()
                        s.end_line, s.complete = no, s.name == name
                        if s.complete:
                            break
                        fail(SectionFormatError, f"Section '{s.name}' (line {s.start_line}) closed implicitly by END of '{name}'")
                else:
                    fail(SectionFormatError, f"Line {no}: END for unknown section '{name}'")
                continue
            if m := P.match('section_start', text):
                close_mpi(no - 1, False)
                sec = Section(m['name'], (m.groupdict().get('title') or '').strip(), start_line=no)
                if stack:
                    sec.path = f'{stack[-1].path}/{sec.name}'
                    stack[-1].children.append(sec)
                else:
                    if self.strict and sec.name in doc.sections:
                        raise SectionFormatError(f"Duplicate section name '{sec.name}'")
                    sec.path = sec.name
                    doc.sections[_unique(doc.sections, sec.name)] = sec
                stack.append(sec)
                ln.kind, ln.section = 'marker', sec.path
                doc.marker_count += 1
                continue
            if P.match('global_json_end', text):
                ln.kind = 'marker'
                fail(ParserError, f'Line {no}: JSON END without start')
                continue
            if m := P.match('global_json_start', text):
                ln.kind = 'marker'
                doc.marker_count += 1
                jb = [m['name'], ln, [], cur]
                continue
            if m := P.match('mpi_end', text):
                ln.kind, name = 'marker', m['name'].strip()
                doc.marker_count += 1
                if mpi and mpi.name == name:
                    close_mpi(no, True)
                else:
                    fail(MPIAllPrintFormatError, f"Line {no}: END '{name}' does not match open MPI print '{mpi.name if mpi else None}'")
                continue
            if m := P.match('mpi_start', text):
                close_mpi(no - 1, False)
                name = m['name'].strip()
                box = cur.mpi_all_prints if cur else doc.mpi_all_prints
                if self.strict and name in box:
                    raise MPIAllPrintFormatError(f"Duplicate MPI ALL PRINT name '{name}'")
                mpi = MPIAllPrint(name, start_line=no)
                box[_unique(box, name)] = mpi
                ln.kind, ln.mpi = 'marker', name
                doc.marker_count += 1
                continue
            if m := P.match('rank_end', text):
                ln.kind = 'marker'
                doc.marker_count += 1
                if mpi:
                    mpi.lines.append(ln)
                if rank and rank.rank == int(m['rank']):
                    close_rank(no, True)
                else:
                    fail(MPIAllPrintFormatError, f"Line {no}: END Rank {m['rank']} does not match an open rank block")
                continue
            if m := P.match('rank_start', text):
                close_rank(no - 1, False)
                rank = RankOutput(int(m['rank']), start_line=no)
                if mpi:
                    if rank.rank in mpi.ranks:
                        fail(MPIAllPrintFormatError, f"Duplicate rank {rank.rank} in MPI ALL PRINT '{mpi.name}'")
                    mpi.ranks[rank.rank] = rank
                    mpi.lines.append(ln)
                ln.kind, ln.rank = 'marker', rank.rank
                doc.marker_count += 1
                continue

            if self.parse_json_lines and text.lstrip()[:1] in ('{', '['):
                try:
                    ln.data, ln.kind = json.loads(text, parse_constant=lambda c: None), 'json'
                except ValueError:
                    pass
            (rank.lines if rank else mpi.lines if mpi else cur.lines if cur else doc.outside).append(ln)
            if rank and mpi:
                mpi.lines.append(ln)

        if jb:
            fail(ParserError, f"JSON block '{jb[0]}' (line {jb[1].no}) is not closed")
        close_mpi(last, False)
        for s in reversed(stack):
            s.end_line = last
            fail(SectionFormatError, f"Section '{s.name}' (line {s.start_line}) is not closed yet (still running?)")
        return doc


def parse(text: str, **kw) -> Document:
    """Parse text -> Document. kw: patterns, strict, strip_ansi, parse_json_lines."""
    return MPIOutputParser(**kw).parse_document(text)


def parse_file(path: Union[str, Path], **kw) -> Document:
    return MPIOutputParser(**kw).parse_document_file(path)


def parse_ccutils_file(path: Union[str, Path], **kw) -> Document:
    """Intuitive alias for `parse_file`; returns sections and timer records."""
    return parse_file(path, **kw)


def parse_ccutils(text: str, **kw) -> Document:
    """Intuitive alias for `parse`; returns a complete parsed Document."""
    return parse(text, **kw)


def parse_ccutils_output_file(path: Path) -> Dict[str, Section]:     # old convenience API
    return parse_file(path).sections


def parse_ccutils_output(text: str) -> Dict[str, Section]:           # old convenience API
    return parse(str(text)).sections


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Parse CCUTILS/MPI output and print a summary or JSON.')
    ap.add_argument('file')
    ap.add_argument('--json', action='store_true', help='dump the parsed structure as JSON')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--lenient', action='store_true', help='treat any ===/---/### banner as a section')
    a = ap.parse_args()
    try:
        d = parse_file(a.file, strict=a.strict,
                       patterns=ParserPatterns.lenient() if a.lenient else None)
    except ParserError as e:
        sys.exit(f'parse error: {e}')
    print(d.to_json(indent=2) if a.json else d.summary())
