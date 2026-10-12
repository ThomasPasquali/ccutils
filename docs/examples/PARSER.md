# Parsing CCUTILS output

The parser extracts named sections, their titles, plain-text content, JSON blocks, MPI rank blocks, and timer records. CMake installation places `ccutils_parser.py` in Python's site-packages, so installed scripts can import it directly. In a source checkout, use `from parser import ...` instead.

```python
from ccutils_parser import parse_file

document = parse_file("run.log")
run = document["run"]
print(run.title)
print(run.json_data)
print(run.raw_text)

# Subsections are child Section objects.
hardware = run.get_child("hardware")
print(hardware.json_data if hardware else None)

# Top-level MPI JSON output is available by rank.
rank_json = document.get_mpi_print("metadata")
print(rank_json.per_rank_json() if rank_json else {})

for record in document.timer_records:
    print(record.record_type, record.fields)
```

For an in-memory string, call `parse(text)` or `parse_ccutils(text)`; both return a `Document`. `parse_timer_records(text)` extracts only timer records from mixed output. `parse_ccutils_output(text)` and `parse_ccutils_output_file(path)` remain available for compatibility and return the top-level section dictionary. `MPIOutputParser` exposes both `parse_document` and the older `parse_string`/`parse_file` methods. Parser exceptions derive from `ParserError`; tolerant mode records malformed structures as warnings, while `strict=True` raises errors.

The installed module also provides a small command-line interface:

```sh
python -m ccutils_parser run.log
python -m ccutils_parser run.log --json

# Visualize the logs in your browser
ccutils-visualize run.log
```
