"""
Tests for the shared output layer: cell formatting, the TSV/CSV writer, the
generated Coot script and the common argparse flags.
"""
import argparse

import pytest

from pdb_python_tools.core import (_format_cell, add_output_args, write_coot_script,
                                   write_table)

HEADER = ["Chain", "Residue", "Distance"]
ROWS = [["A", "10", 1.23456], ["B", "20", 9.5]]


class TestFormatCell:
    def test_rounds_floats_to_precision(self):
        assert _format_cell(1.23456, 2, False) == "1.23"
        assert _format_cell(1.23456, 4, False) == "1.2346"

    def test_zero_precision(self):
        assert _format_cell(1.6, 0, False) == "2"

    def test_pads_to_the_requested_precision(self):
        assert _format_cell(1.5, 3, False) == "1.500"

    def test_full_precision_wins_over_precision(self):
        assert _format_cell(1.23456, 2, True) == "1.23456"

    def test_negative_precision_means_raw(self):
        assert _format_cell(1.23456, -1, False) == "1.23456"

    def test_none_precision_means_raw(self):
        assert _format_cell(1.23456, None, False) == "1.23456"

    @pytest.mark.parametrize("value,expected", [
        ("CA", "CA"),
        (7, "7"),
        ("NA", "NA"),
    ])
    def test_non_floats_pass_through(self, value, expected):
        assert _format_cell(value, 2, False) == expected

    def test_none_prints_as_na(self):
        assert _format_cell(None, 2, False) == "NA"
        assert _format_cell(None, None, True) == "NA"


class TestWriteTable:
    def test_tsv_to_stdout(self, capsys):
        write_table(HEADER, ROWS)
        out = capsys.readouterr().out
        assert out == ("Chain\tResidue\tDistance\n"
                       "A\t10\t1.23\n"
                       "B\t20\t9.50\n")

    def test_csv_to_stdout(self, capsys):
        write_table(HEADER, ROWS, fmt="csv")
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "Chain,Residue,Distance"
        assert out.splitlines()[1] == "A,10,1.23"

    def test_format_is_case_insensitive(self, capsys):
        write_table(HEADER, ROWS, fmt="CSV")
        assert "," in capsys.readouterr().out.splitlines()[0]

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported output format"):
            write_table(HEADER, ROWS, fmt="xlsx")

    def test_writes_to_a_file(self, tmp_path):
        target = tmp_path / "out.tsv"
        write_table(HEADER, ROWS, output=str(target))
        assert target.read_text() == ("Chain\tResidue\tDistance\n"
                                      "A\t10\t1.23\n"
                                      "B\t20\t9.50\n")

    def test_refuses_to_overwrite(self, tmp_path):
        target = tmp_path / "out.tsv"
        target.write_text("keep me\n")
        with pytest.raises(FileExistsError, match="use --force"):
            write_table(HEADER, ROWS, output=str(target))
        assert target.read_text() == "keep me\n"

    def test_force_overwrites(self, tmp_path):
        target = tmp_path / "out.tsv"
        target.write_text("replace me\n")
        write_table(HEADER, ROWS, output=str(target), force=True)
        assert target.read_text().startswith("Chain\t")

    def test_precision_is_honoured(self, tmp_path):
        target = tmp_path / "out.tsv"
        write_table(HEADER, ROWS, output=str(target), precision=4)
        assert "1.2346" in target.read_text()

    def test_full_precision_is_honoured(self, tmp_path):
        target = tmp_path / "out.tsv"
        write_table(HEADER, ROWS, output=str(target), full_precision=True)
        assert "1.23456" in target.read_text()

    def test_unix_line_endings(self, tmp_path):
        target = tmp_path / "out.tsv"
        write_table(HEADER, ROWS, output=str(target))
        assert b"\r\n" not in target.read_bytes()

    def test_csv_quotes_embedded_delimiters(self, tmp_path):
        target = tmp_path / "out.csv"
        write_table(["A"], [["x,y"]], fmt="csv", output=str(target))
        assert target.read_text() == 'A\n"x,y"\n'

    def test_header_only(self, capsys):
        write_table(HEADER, [])
        assert capsys.readouterr().out == "Chain\tResidue\tDistance\n"

    def test_accepts_a_row_generator(self, capsys):
        write_table(["A"], (["%d" % i] for i in range(3)))
        assert capsys.readouterr().out == "A\n0\n1\n2\n"

    def test_comments_are_written_above_the_header(self, capsys):
        write_table(["A"], [["1"]], comments=["first", "second"])
        assert capsys.readouterr().out == "# first\n# second\nA\n1\n"

    def test_no_comments_by_default(self, capsys):
        write_table(["A"], [["1"]])
        assert capsys.readouterr().out == "A\n1\n"

    def test_comments_reach_a_file_too(self, tmp_path):
        target = tmp_path / "out.tsv"
        write_table(["A"], [["1"]], output=str(target), comments=["note"])
        assert target.read_text().splitlines()[0] == "# note"

    def test_comments_are_not_quoted_as_csv_cells(self, tmp_path):
        target = tmp_path / "out.csv"
        write_table(["A"], [["1"]], fmt="csv", output=str(target),
                    comments=["a, b"])
        assert target.read_text().splitlines()[0] == "# a, b"


MARKERS = [
    ("A 10 SER", 1.23456, "Å", 1.111111, 2.222222, 3.333333),
    ("B 20 TYR", 9.5, "Å", -4.0, 5.0, -6.0),
]


class TestWriteCootScript:
    def test_creates_the_script(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "test title", str(target))
        assert target.exists()

    def test_output_is_valid_python(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "test title", str(target))
        compile(target.read_text(), str(target), "exec")

    def test_contains_labels_and_title(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "my title", str(target))
        content = target.read_text()
        assert "my title" in content
        assert "A 10 SER" in content
        assert "B 20 TYR" in content

    def test_value_is_rounded_and_carries_its_unit(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "t", str(target), precision=2)
        assert "1.23 Å" in target.read_text()

    def test_full_precision_value(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "t", str(target), full_precision=True)
        assert "1.23456 Å" in target.read_text()

    def test_blank_unit_is_omitted(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script([("A 1 U", 45.0, "", 0.0, 0.0, 0.0)], "t", str(target))
        assert "'45.00'" in target.read_text()

    def test_coordinates_keep_full_precision(self, tmp_path):
        target = tmp_path / "coot.py"
        # --precision must not degrade the coordinates Coot recentres on
        write_coot_script(MARKERS, "t", str(target), precision=1)
        content = target.read_text()
        assert "1.111111" in content
        assert "2.222222" in content

    def test_marker_order_is_preserved(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "t", str(target))
        content = target.read_text()
        assert content.index("A 10 SER") < content.index("B 20 TYR")

    def test_refuses_to_overwrite(self, tmp_path):
        target = tmp_path / "coot.py"
        target.write_text("keep me\n")
        with pytest.raises(FileExistsError, match="use --force"):
            write_coot_script(MARKERS, "t", str(target))
        assert target.read_text() == "keep me\n"

    def test_force_overwrites(self, tmp_path):
        target = tmp_path / "coot.py"
        target.write_text("replace me\n")
        write_coot_script(MARKERS, "t", str(target), force=True)
        assert "A 10 SER" in target.read_text()

    def test_output_path_is_required(self):
        with pytest.raises(ValueError, match="requires an output path"):
            write_coot_script(MARKERS, "t", None)

    def test_empty_marker_list_still_valid(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script([], "t", str(target))
        compile(target.read_text(), str(target), "exec")

    def test_quotes_in_a_title_are_escaped(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "it's a \"title\"", str(target))
        compile(target.read_text(), str(target), "exec")


class TestAddOutputArgs:
    @pytest.fixture
    def parser(self):
        parser = argparse.ArgumentParser()
        add_output_args(parser)
        return parser

    def test_defaults(self, parser):
        args = parser.parse_args([])
        assert args.format == "tsv"
        assert args.output is None
        assert args.force is False
        assert args.precision == 2
        assert args.full_precision is False
        assert args.coot is None

    def test_short_flags(self, parser):
        args = parser.parse_args(["-f", "csv", "-o", "out.csv"])
        assert args.format == "csv"
        assert args.output == "out.csv"

    def test_long_flags(self, parser):
        args = parser.parse_args(["--format", "csv", "--force", "--precision", "5",
                                  "--full-precision", "--coot", "s.py"])
        assert args.format == "csv"
        assert args.force is True
        assert args.precision == 5
        assert args.full_precision is True
        assert args.coot == "s.py"

    def test_invalid_format_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["-f", "xlsx"])
