"""Tests for the v2 parser (parser/ccutils_run.py). Fast, no compiler or MPI needed:
    python3 -m unittest tests/test_parser.py
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import ccutils  # noqa: E402
import parser as parser_alias  # noqa: E402
from ccutils.cli import main as cli_main  # noqa: E402

GOLDEN = HERE / "golden"
FIXTURES = HERE / "fixtures"


def inline_to_jsonl(text: str) -> str:
    return "\n".join(l[len("@ccutils "):] for l in text.splitlines() if l.startswith("@ccutils "))


class Detect(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(ccutils.detect_format('{"v":2,"type":"data"}\n'), "jsonl")
        self.assertEqual(ccutils.detect_format('hello\n@ccutils {"v":2}\n'), "inline")
        self.assertEqual(ccutils.detect_format("=+=+=+= a :: A =+=+=+=\n"), "legacy")
        self.assertEqual(ccutils.detect_format(""), "legacy")

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ccutils.FormatError):
            ccutils.loads("x", format="xml")

    def test_parser_name_remains_a_compatibility_alias(self):
        self.assertIs(parser_alias.load, ccutils.load)


class JsonlModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parsed = ccutils.load(GOLDEN / "multifile.default.machine.jsonl")

    def test_structure_and_titles(self):
        run = self.parsed
        self.assertEqual(run.format, "jsonl")
        self.assertEqual([s.path for s in run.sections()], ["run", "run/solve"])
        self.assertEqual(run["run"].title, "Whole run")
        self.assertTrue(run["run"].complete and run["run/solve"].complete)
        self.assertEqual(run["run"].child("solve"), run["run/solve"])
        self.assertEqual(run["run"]["solve"], run["run/solve"])

    def test_abnormal_inner_section_close_and_warning(self):
        records = [
            {"v": 2, "type": "run_start", "section": None, "scope": "global", "rank": None,
             "ccutils_version": "1.0.0", "format": "legacy", "host": "test", "seq": 1},
            {"v": 2, "type": "section_start", "section": "outer", "scope": "global", "rank": None,
             "title": "Outer", "seq": 2},
            {"v": 2, "type": "section_start", "section": "outer/inner", "scope": "global", "rank": None,
             "title": "Inner", "seq": 3},
            {"v": 2, "type": "section_end", "section": "outer/inner", "scope": "global", "rank": None,
             "ok": False, "seq": 4},
            {"v": 2, "type": "warning", "section": "outer/inner", "scope": "global", "rank": None,
             "message": "inner closed implicitly", "seq": 5},
            {"v": 2, "type": "section_end", "section": "outer", "scope": "global", "rank": None,
             "ok": True, "seq": 6},
        ]
        run = ccutils.loads("\n".join(json.dumps(record) for record in records) + "\n")
        self.assertFalse(run["outer/inner"].ok)
        self.assertTrue(run["outer"].ok)
        self.assertIn("inner closed implicitly", run.warnings)
        self.assertEqual(run.unknown_types, {})

    def test_data_and_timers(self):
        run = self.parsed
        self.assertEqual(run["run/solve"].data, {"residual": 0.25})
        self.assertEqual(run["run"].data["iterations"], 4)
        self.assertEqual(run["run"].data["report"], {"kernel_samples": 5})
        self.assertEqual(run["run"].timers["kernel"].n, 5)
        self.assertEqual(run["run"].timers["kernel"].last.scope, "rank")
        self.assertEqual(run["run"].timers["kernel"].last.unit, "ms")

    def test_lookup_errors_suggest_alternatives(self):
        with self.assertRaises(KeyError) as ctx:
            self.parsed["run/solv"]
        self.assertIn("run/solve", str(ctx.exception))
        self.assertIn("run/solve", self.parsed)
        self.assertEqual([s.path for s in self.parsed.find("solve")], ["run/solve"])

    def test_tables(self):
        timers = self.parsed.timers_table()
        self.assertEqual({r["name"] for r in timers}, {"kernel", "total"})
        self.assertEqual({r["scope"] for r in timers}, {"rank"})
        data = {(r["section"], r["key"]): r["value"] for r in self.parsed.data_table()}
        self.assertEqual(data[("run/solve", "residual")], 0.25)

    def test_dataframes_when_pandas_is_available(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")
        df = self.parsed.timers_df()
        self.assertEqual(set(df["name"]), {"kernel", "total"})
        self.assertEqual(len(self.parsed.data_df()), len(self.parsed.data_table()))

    def test_tree_lists_everything(self):
        text = self.parsed.tree(file=False)
        for expected in ("run   Whole run", "solve   Solve", "timers: kernel, total", "data: residual"):
            self.assertIn(expected, text)


class SameModelFromEveryFormat(unittest.TestCase):
    """One run, recorded both as human log and as machine records, parsed three ways."""

    @classmethod
    def setUpClass(cls):
        text = (FIXTURES / "multifile.inline.txt").read_text()
        cls.inline = ccutils.loads(text, "inline")
        cls.legacy = ccutils.loads(text, "legacy")
        cls.jsonl = ccutils.loads(inline_to_jsonl(text), "jsonl")

    def test_sections_and_data_are_identical(self):
        for other in (self.inline, self.legacy):
            self.assertEqual([s.path for s in other.sections()], [s.path for s in self.jsonl.sections()])
            for a, b in zip(self.jsonl.sections(), other.sections()):
                self.assertEqual(a.title, b.title)
                self.assertEqual(a.data, b.data)
                self.assertEqual(a.complete, b.complete)

    def test_timers_agree_within_the_legacy_rounding(self):
        for other in (self.inline, self.legacy):
            for a, b in zip(self.jsonl.timers_table(), other.timers_table()):
                self.assertEqual((a["section"], a["name"], a["scope"], a["n"]),
                                 (b["section"], b["name"], b["scope"], b["n"]))
                for key in ("avg", "sd", "min", "max", "sum"):
                    self.assertAlmostEqual(a[key], b[key], delta=5.1e-7)

    def test_lossy_is_reported_only_for_the_legacy_log(self):
        self.assertTrue(self.legacy.lossy)
        self.assertFalse(self.inline.lossy)
        self.assertFalse(self.legacy.check() or self.inline.check())

    def test_auto_detection_picks_the_right_front_end(self):
        text = (FIXTURES / "multifile.inline.txt").read_text()
        self.assertEqual(ccutils.loads(text).format, "inline")
        self.assertEqual(ccutils.loads("\n".join(l for l in text.splitlines()
                                                 if not l.startswith("@ccutils "))).format, "legacy")


class MpiRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = (FIXTURES / "multifile_mpi.inline.txt").read_text()
        cls.machine = ccutils.loads(text, "inline")
        cls.legacy = ccutils.loads(text, "legacy")

    def test_communicator_timer(self):
        halo = self.machine["run"].timers["halo"]
        self.assertEqual(halo.comm.scope, "comm")
        self.assertEqual((halo.n, halo.comm.comm["size"], halo.comm.comm["root"]), (6, 2, 0))
        self.assertIsNone(halo.by_rank().get(0))
        self.assertEqual(self.legacy["run"].timers["halo"].n, 6)   # same pooled report

    def test_machine_stream_has_global_data_only_for_now(self):
        self.assertEqual(self.machine["run"].data, {"iterations": 3})
        self.assertEqual(self.machine["run"].rank_data, {})

    def test_legacy_log_also_gives_rank_data_and_text(self):
        run = self.legacy
        self.assertEqual(run["run"].data, {"iterations": 3})
        self.assertEqual(run["run"].per_rank("rank"), {0: 0, 1: 1})
        self.assertEqual(run["run/work"].per_rank("samples"), {0: 3, 1: 3})
        worker = run["run/work"].prints["worker"][0]
        self.assertEqual(worker.ranks[1].lines, ["hello from worker, rank 1"])
        self.assertEqual(run["run"].prints["stats"][0].ranks[0].json(), [{"rank": 0}])


class Sidecars(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        shutil.copy(GOLDEN / "mpi.default.np2.machine.jsonl", self.tmp / "m.jsonl")
        shutil.copy(GOLDEN / "mpi.default.np2.machine.r1.jsonl", self.tmp / "m.jsonl.r1")

    def test_sidecar_of_another_rank_is_merged(self):
        run = ccutils.load(self.tmp / "m.jsonl")
        self.assertEqual([Path(s).name for s in run.sources], ["m.jsonl", "m.jsonl.r1"])
        report = run.outside.timers["root_last"].comm
        self.assertEqual((report.comm["root"], report.scope), (1, "comm"))
        self.assertEqual(run.check(), [])           # each file has its own seq counter

    def test_sidecars_can_be_disabled(self):
        run = ccutils.load(self.tmp / "m.jsonl", sidecars=False)
        self.assertNotIn("root_last", run.outside.timers)
        self.assertEqual(len(run.sources), 1)


class Robustness(unittest.TestCase):
    def lines(self):
        return (GOLDEN / "multifile.default.machine.jsonl").read_text().splitlines()

    def test_truncated_log_is_reported(self):
        run = ccutils.loads("\n".join(self.lines()[:-2]), "jsonl")
        problems = run.check()
        self.assertTrue(any("never closed" in p for p in problems), problems)
        self.assertIn("[never closed]", run.tree(file=False))

    def test_lost_line_is_reported(self):
        lines = self.lines()
        del lines[4]
        run = ccutils.loads("\n".join(lines), "jsonl")
        self.assertTrue(any("seq" in w for w in run.warnings), run.warnings)

    def test_garbage_unknown_types_and_newer_schema(self):
        text = "\n".join(self.lines() + ["this is not json", '{"v":9,"type":"hologram","seq":13}',
                                          '{"v":9,"type":"data","key":"k","value":1,"seq":14}'])
        run = ccutils.loads(text, "jsonl")
        self.assertTrue(any("not valid JSON" in w for w in run.warnings))
        self.assertEqual(run.unknown_types, {"hologram": 1})
        self.assertTrue(any("newer" in w for w in run.warnings))
        self.assertEqual(run.outside.data, {"k": 1})
        self.assertEqual(run["run"].timers["kernel"].n, 5)   # the good records are unaffected

    def test_record_before_its_section_is_kept(self):
        run = ccutils.loads('{"v":2,"type":"data","section":"a/b","scope":"global","key":"x","value":1}', "jsonl")
        self.assertEqual(run["a/b"].data, {"x": 1})
        self.assertTrue(run["a/b"].implicit)
        self.assertTrue(any("before it was opened" in w for w in run.warnings))

    def test_section_reopened_in_a_loop_appears_several_times(self):
        rec = lambda t, **k: json.dumps({"v": 2, "type": t, "section": "loop", "scope": "global", **k})
        text = "\n".join([rec("section_start", title="L"), rec("data", key="i", value=0), rec("section_end", ok=True),
                          rec("section_start", title="L"), rec("data", key="i", value=1), rec("section_end", ok=True)])
        run = ccutils.loads(text, "jsonl")
        self.assertEqual([s.data["i"] for s in run.all("loop")], [0, 1])
        self.assertEqual(run["loop"].data["i"], 0)

    def test_utf8_and_escapes_survive(self):
        rec = {"v": 2, "type": "data", "section": None, "scope": "global", "key": "k",
               "value": "caf\u00e9 \"q\"\n\u20ac"}
        self.assertEqual(ccutils.loads(json.dumps(rec), "jsonl").outside.data["k"], rec["value"])

    def test_text_record_lines_and_base64_round_trip(self):
        records = [
            {"v": 2, "type": "text", "section": None, "scope": "rank", "rank": 1,
             "print": "p", "call": 1, "part": 0, "last": True,
             "lines": ["one", "", "two\r"], "trailing_newline": False},
            {"v": 2, "type": "text", "section": None, "scope": "rank", "rank": 2,
             "print": "p", "call": 1, "part": 0, "last": True,
             "encoding": "base64", "data": "YQBi", "trailing_newline": False},
        ]
        run = ccutils.loads("\n".join(json.dumps(r) for r in records), "jsonl")
        block = run.outside.prints["p"][0]
        self.assertEqual(block.ranks[1].text, "one\n\ntwo\r")
        self.assertEqual(block.ranks[2].text, "a\x00b")

    def test_legacy_rank_text_splits_only_on_lf(self):
        text = (FIXTURES / "multifile_mpi.inline.txt").read_text()
        content = "hello\fform\u2028worker\vwith\x1ccontrols, rank 1"
        text = text.replace("hello from worker, rank 1", content)
        run = ccutils.loads(text, "legacy")
        self.assertEqual(run["run/work"].prints["worker"][0].ranks[1].lines, [content])


class CommandLine(unittest.TestCase):
    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli_main(list(args))
        return code, out.getvalue()

    def test_commands(self):
        path = str(GOLDEN / "multifile.default.machine.jsonl")
        code, out = self.run_cli("tree", path)
        self.assertEqual(code, 0)
        self.assertIn("solve   Solve", out)
        code, out = self.run_cli("timers", path, "--csv")
        self.assertTrue(out.startswith("section,name,group,scope,rank,n,"))
        code, out = self.run_cli("data", path, "--section", "run/solve", "--key", "residual")
        self.assertIn("0.25", out)
        self.assertEqual(self.run_cli("check", path)[0], 0)

    def test_check_fails_on_a_truncated_file(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        lines = (GOLDEN / "multifile.default.machine.jsonl").read_text().splitlines()
        (tmp / "t.jsonl").write_text("\n".join(lines[:-2]) + "\n")
        code, out = self.run_cli("check", str(tmp / "t.jsonl"))
        self.assertEqual(code, 1)
        self.assertIn("never closed", out)


if __name__ == "__main__":
    unittest.main()
