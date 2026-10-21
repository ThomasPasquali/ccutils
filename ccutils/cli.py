"""Command line interface for inspecting CCUTILS run logs."""

import argparse
import csv
import sys

from .ccutils_run import load


def _print_rows(rows, as_csv):
    if not rows:
        print("(none)")
        return
    columns = list(rows[0])
    if as_csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in columns))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ccutils", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("tree", "timers", "data", "check"):
        command = sub.add_parser(name)
        command.add_argument("file")
        command.add_argument("--no-sidecars", action="store_true",
                             help="do not merge FILE.r<N>")
        if name in ("timers", "data"):
            command.add_argument("--csv", action="store_true")
        if name == "data":
            command.add_argument("--section")
            command.add_argument("--key")
    args = ap.parse_args(argv)
    run = load(args.file, sidecars=not args.no_sidecars)
    if args.command == "tree":
        run.tree()
    elif args.command == "timers":
        _print_rows(run.timers_table(), args.csv)
    elif args.command == "data":
        rows = [row for row in run.data_table()
                if (args.section is None or row["section"] == args.section)
                and (args.key is None or row["key"] == args.key)]
        _print_rows(rows, args.csv)
    else:
        problems = run.check()
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)")
        return 1 if problems else 0
    return 0
