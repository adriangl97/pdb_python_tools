"""
Tests for the shared output layer: cell formatting, the TSV/CSV writer, the
generated Coot script and the common argparse flags.
"""
import argparse
import sys
import types

import pytest

from pdb_python_tools.core import (COOT_GRAPH_BANDS, _format_cell, add_output_args,
                                   write_coot_script, write_table)

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


def run_generated_script(tmp_path, markers=None):
    """
    Run a generated script the way Coot would, and hand back its namespace.

    Outside Coot it only prints its list, so what is left to look at is how it
    would have found Coot and GTK.
    """
    target = tmp_path / "coot.py"
    write_coot_script(MARKERS if markers is None else markers, "t", str(target),
                      force=True)
    namespace = {"__name__": "not_main"}
    exec(compile(target.read_text(), str(target), "exec"), namespace)
    return namespace


@pytest.fixture
def without_gtk(monkeypatch):
    """A process with neither PyGTK nor PyGObject, whatever the test machine has."""
    for name in ("gtk", "gi"):
        monkeypatch.setitem(sys.modules, name, None)


class TestGeneratedScriptFindsCoot:
    """
    The script has to work in Coot 0.9 and Coot 1, which agree on nothing:
    Python 2 and PyGTK on one side, Python 3 and PyGObject on the other.
    """

    def test_coot_09_keeps_its_functions_in_the_scripts_namespace(self, tmp_path,
                                                                  without_gtk):
        namespace = run_generated_script(tmp_path)
        centred = []
        namespace["set_rotation_centre"] = lambda x, y, z: centred.append((x, y, z))
        namespace["_recentre_function"]()(1.0, 2.0, 3.0)
        assert centred == [(1.0, 2.0, 3.0)]

    def test_coot_1_keeps_them_in_the_coot_module(self, tmp_path, monkeypatch,
                                                 without_gtk):
        coot = types.ModuleType("coot")
        centred = []
        coot.set_rotation_centre = lambda x, y, z: centred.append((x, y, z))
        monkeypatch.setitem(sys.modules, "coot", coot)
        namespace = run_generated_script(tmp_path)
        namespace["_recentre_function"]()(1.0, 2.0, 3.0)
        assert centred == [(1.0, 2.0, 3.0)]

    def test_no_coot_no_recentring(self, tmp_path, monkeypatch, without_gtk):
        monkeypatch.setitem(sys.modules, "coot", None)
        namespace = run_generated_script(tmp_path)
        assert namespace["_recentre_function"]() is None

    def test_the_list_is_printed_when_there_is_no_coot(self, tmp_path, capsys,
                                                      without_gtk):
        run_generated_script(tmp_path)
        assert "A 10 SER" in capsys.readouterr().out

    def test_pygtk_is_recognised(self, tmp_path, monkeypatch):
        gtk = types.ModuleType("gtk")
        gtk.pygtk_version = (2, 24, 0)
        monkeypatch.setitem(sys.modules, "gtk", gtk)
        namespace = run_generated_script(tmp_path)
        assert namespace["_gtk"]() == (gtk, "pygtk")

    def test_pygobjects_stand_in_gtk_is_not_pygtk(self, tmp_path, monkeypatch):
        # PyGObject installs a "gtk" module that imports and then raises on every
        # attribute; taking it for PyGTK is how a Coot 1 run used to break
        monkeypatch.setitem(sys.modules, "gtk", types.ModuleType("gtk"))
        monkeypatch.setitem(sys.modules, "gi", None)
        namespace = run_generated_script(tmp_path)
        assert namespace["_gtk"]() == (None, None)


SERIES = ("Max displacement (Å)", "Average displacement (Å)",
          "CA/C1' displacement (Å)")

# (label, chain, seqid, values per series, unit, x, y, z); A 12 GLY has no
# CA/C1' value, the way a residue missing from one structure has none
GRAPH = [
    ("A 10 SER", "A", "10", (3.5, 2.0, 1.0), "Å", 1.0, 2.0, 3.0),
    ("A 12 GLY", "A", "12A", (0.3, 0.2, None), "Å", 1.1, 2.1, 3.1),
    ("A 40 LYS", "A", "40", (0.75, 0.5, 0.25), "Å", 1.2, 2.2, 3.2),
    ("B 20 TYR", "B", "20", (1.5, 1.0, 0.5), "Å", 4.0, 5.0, 6.0),
]


def graph_script(tmp_path, graph=None, **kwargs):
    """The text of a generated script that carries a graph."""
    target = tmp_path / "coot.py"
    kwargs.setdefault("graph_series", SERIES)
    write_coot_script(MARKERS, "t", str(target), force=True,
                      graph=GRAPH if graph is None else graph, **kwargs)
    return target.read_text()


def graph_namespace(tmp_path, graph=None, **kwargs):
    """The namespace of such a script, run the way Coot would run it."""
    text = graph_script(tmp_path, graph, **kwargs)
    namespace = {"__name__": "not_main"}
    exec(compile(text, str(tmp_path / "coot.py"), "exec"), namespace)
    return namespace


class FakeCairo:
    """
    Stands in for the cairo context Coot's GTK hands the drawing code.

    It keeps the filled rectangles and the text that was drawn, and measures
    every character as five points wide, which is close enough for the
    geometry to be checked without a display.
    """

    def __init__(self):
        self.rectangles = []
        self.texts = []
        self.placed = []
        self.strokes = 0
        self._colour = None
        self._pending = None
        self._at = (0.0, 0.0)

    def set_source_rgb(self, red, green, blue):
        self._colour = (round(red, 3), round(green, 3), round(blue, 3))

    def rectangle(self, x, y, width, height):
        self._pending = (x, y, width, height)

    def fill(self):
        self.rectangles.append((self._pending, self._colour))

    def move_to(self, x, y):
        self._at = (x, y)

    def line_to(self, x, y):
        pass

    def stroke(self):
        self.strokes += 1

    def show_text(self, text):
        self.texts.append(text)
        self.placed.append((self._at, text))

    def text_extents(self, text):
        return (0.0, 0.0, len(text) * 5.0, 8.0, len(text) * 5.0, 0.0)

    def set_line_width(self, width):
        pass

    def select_font_face(self, family):
        pass

    def set_font_size(self, size):
        pass

    @property
    def bars(self):
        """The filled rectangles, without the background the draw starts with."""
        return self.rectangles[1:]

    @property
    def gridlines(self):
        """How many horizontal rules were drawn, leaving out the x-axis baseline."""
        return self.strokes - 1


def fake_areas(count=2):
    """Stand-ins for the graph window's drawing areas, counting their redraws."""
    areas = []
    for _index in range(count):
        area = types.SimpleNamespace(redraws=0, sizes=[])
        area.queue_draw = (lambda a: lambda: setattr(a, "redraws",
                                                     a.redraws + 1))(area)
        area.set_size_request = (lambda a: lambda w, h: a.sizes.append((w, h)))(area)
        areas.append((area, None))
    return areas


def drawn(namespace, chain_index=0):
    """Draw one chain's graph and hand back the graph and what it drew."""
    chain, entries = namespace["_graph_chains"]()[chain_index]
    graph = namespace["_ChainGraph"](chain, entries)
    cr = FakeCairo()
    graph.draw(cr, graph.width(), namespace["GRAPH_HEIGHT"])
    return graph, cr


class TestCootScriptGraphData:
    def test_the_graph_reaches_the_script(self, tmp_path):
        content = graph_script(tmp_path)
        assert "'A 10 SER', 'A', 10, (3.5, 2.0, 1.0)" in content
        assert "'B 20 TYR', 'B', 20, (1.5, 1.0, 0.5)" in content

    def test_an_insertion_code_keeps_its_residue_number(self, tmp_path):
        assert "'A 12 GLY', 'A', 12," in graph_script(tmp_path)

    def test_a_seqid_without_a_number_is_left_out(self, tmp_path):
        graph = [("X ? UNK", "X", ".", (1.0, 1.0, 1.0), "Å", 0.0, 0.0, 0.0)]
        assert graph_namespace(tmp_path, graph)["GRAPH"] == []

    def test_no_graph_leaves_it_empty(self, tmp_path):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "t", str(target), force=True)
        namespace = {"__name__": "not_main"}
        exec(compile(target.read_text(), str(target), "exec"), namespace)
        assert namespace["GRAPH"] == []

    def test_the_value_text_follows_precision(self, tmp_path):
        assert "'3.500 Å'" in graph_script(tmp_path, precision=3)

    def test_a_missing_value_is_kept_as_none(self, tmp_path):
        entry = graph_namespace(tmp_path)["GRAPH"][1]
        assert entry[3] == (0.3, 0.2, None)
        assert entry[4] == ("0.30 Å", "0.20 Å", "NA")

    def test_graph_coordinates_keep_full_precision(self, tmp_path):
        graph = [("A 1 SER", "A", "1", (1.0,), "Å", 1.111111, 2.222222, 3.333333)]
        content = graph_script(tmp_path, graph, graph_series=("Only",), precision=1)
        assert "1.111111" in content and "2.222222" in content

    def test_a_graph_without_series_is_refused(self, tmp_path):
        target = tmp_path / "coot.py"
        with pytest.raises(ValueError, match="graph_series"):
            write_coot_script(MARKERS, "t", str(target), force=True, graph=GRAPH)

    def test_the_default_bands_are_written(self, tmp_path):
        assert graph_namespace(tmp_path)["GRAPH_BANDS"] == list(COOT_GRAPH_BANDS)

    def test_the_bands_can_be_replaced(self, tmp_path):
        bands = ((1.0, "#000000"), (None, "#ffffff"))
        namespace = graph_namespace(tmp_path, graph_bands=bands)
        assert namespace["GRAPH_BANDS"] == [(1.0, "#000000"), (None, "#ffffff")]

    def test_the_graph_window_gets_its_own_title(self, tmp_path):
        namespace = graph_namespace(tmp_path, graph_title="displacement")
        assert namespace["GRAPH_TITLE"] == "displacement"

    def test_the_series_reach_the_script(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert namespace["GRAPH_SERIES"] == list(SERIES)
        assert namespace["GRAPH_SELECTED"] == [0]

    def test_the_window_can_open_on_another_series(self, tmp_path):
        namespace = graph_namespace(tmp_path, graph_selected=1)
        assert namespace["GRAPH_SELECTED"] == [1]
        assert namespace["_entry_value"](namespace["GRAPH"][0]) == 2.0

    def test_a_series_that_does_not_exist_is_refused(self, tmp_path):
        target = tmp_path / "coot.py"
        with pytest.raises(ValueError, match="graph_selected"):
            write_coot_script(MARKERS, "t", str(target), force=True, graph=GRAPH,
                              graph_series=SERIES, graph_selected=3)

    def test_the_graph_title_falls_back_to_the_dialog_title(self, tmp_path):
        assert graph_namespace(tmp_path)["GRAPH_TITLE"] == "t"


class TestGeneratedGraph:
    """The chart the script draws for itself, without a GTK to draw it on."""

    def test_a_value_takes_the_colour_of_its_band(self, tmp_path):
        colour = graph_namespace(tmp_path)["_band_colour"]
        assert colour(0.2) == "#55dd55"
        assert colour(0.5) == "#eecc22"
        assert colour(1.0) == "#ee9933"
        assert colour(2.0) == "#dd4444"
        assert colour(500.0) == "#dd4444"

    def test_chains_are_grouped_in_the_order_they_appear(self, tmp_path):
        chains = graph_namespace(tmp_path)["_graph_chains"]()
        assert [chain for chain, _entries in chains] == ["A", "B"]
        assert [len(entries) for _chain, entries in chains] == [3, 1]

    def test_each_chain_gets_its_own_rounded_up_scale(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        chains = namespace["_graph_chains"]()
        scales = [namespace["_ChainGraph"](chain, entries).ymax()
                  for chain, entries in chains]
        # chain A tops out at 3.5 and chain B at 1.5, each rounded up
        assert scales == [4.0, 2.0]

    def test_one_bar_per_residue_in_its_band_colour(self, tmp_path):
        graph, cr = drawn(graph_namespace(tmp_path))
        assert len(cr.bars) == 3
        # 3.50 is red, 0.30 green, 0.75 yellow, in the order of the entries
        assert [colour for _rect, colour in cr.bars] == [
            (0.867, 0.267, 0.267), (0.333, 0.867, 0.333), (0.933, 0.8, 0.133)]

    def test_a_bar_is_as_tall_as_its_share_of_the_scale(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        graph, cr = drawn(namespace)
        (_x, _y, _width, tallest), _colour = cr.bars[0]
        (_x, _y, _width, shortest), _colour = cr.bars[1]
        # 3.5 and 0.3 against a scale of 4.0
        assert round(tallest / shortest, 2) == round(3.5 / 0.3, 2)

    def test_the_axes_are_labelled(self, tmp_path):
        graph, cr = drawn(graph_namespace(tmp_path))
        # the scale first, then the residue numbers of chain A (10 to 40)
        assert cr.texts[:5] == ["0", "1", "2", "3", "4"]
        assert cr.texts[5:] == ["10", "20", "30", "40"]

    def test_a_click_finds_the_residue_under_it(self, tmp_path):
        graph, _cr = drawn(graph_namespace(tmp_path))
        start, end, entry = graph.bars[0]
        assert graph.clicked((start + end) / 2.0)[0] == "A 10 SER"

    def test_a_click_beside_every_bar_finds_nothing(self, tmp_path):
        graph, _cr = drawn(graph_namespace(tmp_path))
        _start, last_end, _entry = graph.bars[-1]
        assert graph.clicked(last_end + 50.0) is None

    def test_a_click_recentres_and_names_what_was_clicked(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        graph, _cr = drawn(namespace)
        centred = []
        status = types.SimpleNamespace(text="")
        status.set_text = lambda text: setattr(status, "text", text)
        pick = namespace["_make_pick"](
            graph, lambda x, y, z: centred.append((x, y, z)), status)
        start, end, _entry = graph.bars[0]
        pick((start + end) / 2.0)
        assert centred == [(1.0, 2.0, 3.0)]
        assert status.text == "A 10 SER    3.50 Å"

    def test_a_click_on_nothing_leaves_the_view_alone(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        graph, _cr = drawn(namespace)
        centred = []
        status = types.SimpleNamespace(text="")
        status.set_text = lambda text: setattr(status, "text", text)
        pick = namespace["_make_pick"](
            graph, lambda x, y, z: centred.append((x, y, z)), status)
        _start, last_end, _entry = graph.bars[-1]
        pick(last_end + 50.0)
        assert centred == []

    def test_a_chain_with_one_residue_still_draws_a_bar(self, tmp_path):
        graph, cr = drawn(graph_namespace(tmp_path), chain_index=1)
        assert len(cr.bars) == 1
        # and not one slab across the whole graph
        (_x, _y, width, _height), _colour = cr.bars[0]
        assert width == 7.0

    def test_the_legend_names_every_band(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        cr = FakeCairo()
        namespace["_draw_legend"](cr, 560, 22)
        assert cr.texts == ["< 0.5", "0.5 - 1", "1 - 2", "> 2"]
        assert len(cr.bars) == len(COOT_GRAPH_BANDS)

    def test_no_window_without_a_graph(self, tmp_path, without_gtk):
        target = tmp_path / "coot.py"
        write_coot_script(MARKERS, "t", str(target), force=True)
        namespace = {"__name__": "not_main"}
        exec(compile(target.read_text(), str(target), "exec"), namespace)
        assert namespace["show_coot_graph"]() is None

    def test_no_window_outside_coot(self, tmp_path, without_gtk):
        assert graph_namespace(tmp_path)["show_coot_graph"]() is None


class TestConstantBarWidth:
    """A bar is the same width whatever the chain it belongs to."""

    def wide_graph(self, tmp_path, last):
        """One chain of two residues, `last` residue numbers apart."""
        graph = [("A 1 SER", "A", "1", (1.0,), "Å", 0.0, 0.0, 0.0),
                 ("A %d LYS" % last, "A", str(last), (2.0,), "Å", 1.0, 1.0, 1.0)]
        return graph_namespace(tmp_path, graph, graph_series=("Value (Å)",))

    def test_a_short_and_a_long_chain_get_the_same_bars(self, tmp_path):
        short = self.wide_graph(tmp_path, 12)
        long = self.wide_graph(tmp_path, 400)
        widths = []
        for namespace in (short, long):
            _graph, cr = drawn(namespace)
            widths.extend([rect[2] for rect, _colour in cr.bars])
        assert widths == [7.0, 7.0, 7.0, 7.0]

    def test_a_longer_chain_gets_a_wider_graph(self, tmp_path):
        widths = []
        for last in (100, 400):
            namespace = self.wide_graph(tmp_path, last)
            chain, entries = namespace["_graph_chains"]()[0]
            widths.append(namespace["_ChainGraph"](chain, entries).width())
        # 9 points per residue either way, so the canvas grows with the chain
        assert widths[1] - widths[0] == (400 - 100) * 9

    def test_residues_sit_nine_points_apart(self, tmp_path):
        namespace = self.wide_graph(tmp_path, 12)
        graph, _cr = drawn(namespace)
        starts = [start for start, _end, _entry in graph.bars]
        assert round(starts[1] - starts[0], 6) == 11 * 9.0

    def test_a_chain_too_long_for_the_canvas_shrinks_to_fit(self, tmp_path):
        namespace = self.wide_graph(tmp_path, 40000)
        graph, cr = drawn(namespace)
        assert graph.width() == namespace["GRAPH_MAX_WIDTH"][0]
        widths = [rect[2] for rect, _colour in cr.bars]
        # narrower than the constant, but still on the canvas
        assert 0 < widths[0] < 7.0
        assert graph.bars[-1][1] <= graph.width()

    def test_a_squeezed_chain_says_so(self, tmp_path):
        namespace = self.wide_graph(tmp_path, 40000)
        _graph, cr = drawn(namespace)
        assert any("too many residues" in text for text in cr.texts)

    def test_a_chain_that_fits_says_nothing(self, tmp_path):
        namespace = self.wide_graph(tmp_path, 400)
        _graph, cr = drawn(namespace)
        assert not any("too many residues" in text for text in cr.texts)


class TestXAxisTicks:
    """The x axis is numbered every ten residues."""

    def chain_of(self, tmp_path, first, last):
        """One chain with a residue at each end of the range."""
        graph = [("A %d SER" % first, "A", str(first), (1.0,), "Å", 0.0, 0.0, 0.0),
                 ("A %d LYS" % last, "A", str(last), (2.0,), "Å", 1.0, 1.0, 1.0)]
        namespace = graph_namespace(tmp_path, graph, graph_series=("Value (Å)",))
        _graph, cr = drawn(namespace)
        # the residue numbers are the row of text furthest down; the scale is
        # drawn higher up, against the gridlines
        lowest = max(y for (_x, y), _text in cr.placed)
        return [text for (_x, y), text in cr.placed if y == lowest]

    def test_every_ten_residues(self, tmp_path):
        assert self.chain_of(tmp_path, 1, 42) == ["10", "20", "30", "40"]

    def test_ticks_land_on_round_tens(self, tmp_path):
        assert self.chain_of(tmp_path, 47, 73) == ["50", "60", "70"]

    def test_a_long_chain_keeps_the_ten(self, tmp_path):
        assert self.chain_of(tmp_path, 400, 460) == [
            "400", "410", "420", "430", "440", "450", "460"]

    def test_a_chain_shorter_than_ten_is_still_placed(self, tmp_path):
        # nothing round falls inside 844-848, so the first residue is named
        assert self.chain_of(tmp_path, 844, 848) == ["844"]

    def test_a_squeezed_chain_falls_back_to_a_wider_step(self, tmp_path):
        # at GRAPH_MAX_WIDTH this chain has far less than a point per residue,
        # so tens would be unreadable and the axis steps by more
        labels = self.chain_of(tmp_path, 1, 40000)
        gaps = set(int(later) - int(earlier)
                   for earlier, later in zip(labels, labels[1:]))
        assert gaps and all(gap % 10 == 0 for gap in gaps)
        assert min(gaps) > 10


class TestYAxis:
    """Each chain scales to its own maximum, with a gridline every angstrom."""

    def chain_reaching(self, tmp_path, *values):
        """One chain of consecutive residues holding `values`."""
        graph = [("A %d SER" % (index + 1), "A", str(index + 1), (value,), "Å",
                  0.0, 0.0, 0.0) for index, value in enumerate(values)]
        return graph_namespace(tmp_path, graph, graph_series=("Value (Å)",))

    def scale_labels(self, namespace):
        """The numbers up the y axis, and how many gridlines were drawn."""
        _graph, cr = drawn(namespace)
        lowest = max(y for (_x, y), _text in cr.placed)
        labels = [text for (_x, y), text in cr.placed if y != lowest]
        return labels, cr.gridlines

    def test_the_scale_stops_at_the_next_whole_angstrom(self, tmp_path):
        for highest, expected in ((3.5, 4.0), (2.0, 2.0), (0.3, 1.0), (7.1, 8.0)):
            namespace = self.chain_reaching(tmp_path, highest, 0.1)
            chain, entries = namespace["_graph_chains"]()[0]
            assert namespace["_ChainGraph"](chain, entries).ymax() == expected

    def test_a_gridline_every_angstrom(self, tmp_path):
        labels, gridlines = self.scale_labels(self.chain_reaching(tmp_path, 4.2))
        assert labels == ["0", "1", "2", "3", "4", "5"]
        assert gridlines == 6

    def test_a_quiet_chain_still_gets_a_scale(self, tmp_path):
        labels, gridlines = self.scale_labels(self.chain_reaching(tmp_path, 0.05))
        assert labels == ["0", "1"]
        assert gridlines == 2

    def test_a_chain_with_no_values_at_all_still_draws(self, tmp_path):
        namespace = self.chain_reaching(tmp_path, None, None)
        labels, _gridlines = self.scale_labels(namespace)
        _graph, cr = drawn(namespace)
        assert labels == ["0", "1"]
        assert cr.bars == []

    def test_far_apart_chains_are_scaled_apart(self, tmp_path):
        # the same bar height means different things in two chains now
        namespace = graph_namespace(tmp_path)
        heights = []
        for index in (0, 1):
            _graph, cr = drawn(namespace, chain_index=index)
            heights.append(cr.bars[0][0][3])
        # chain A's 3.5 of 4.0 and chain B's 1.5 of 2.0 are both most of the way up
        assert 0.8 < heights[0] / heights[1] < 1.2

    def test_a_tall_chain_keeps_its_gridlines_but_thins_the_numbers(self, tmp_path):
        labels, gridlines = self.scale_labels(self.chain_reaching(tmp_path, 39.0))
        assert gridlines == 40                     # still one per angstrom
        assert labels == ["0", "5", "10", "15", "20", "25", "30", "35"]
    """The "Value..." button draws the bars from a different column."""

    def test_the_first_series_is_shown_to_start_with(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert namespace["_selected"]() == 0
        assert namespace["_entry_value"](namespace["GRAPH"][0]) == 3.5

    def test_choosing_a_series_changes_the_bars(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        _graph, cr = drawn(namespace)
        tallest_max = cr.bars[0][0][3]
        namespace["GRAPH_SELECTED"][0] = 1              # average displacement
        _graph, cr = drawn(namespace)
        assert namespace["_entry_value"](namespace["GRAPH"][0]) == 2.0
        # 3.5 of 4.0 against 2.0 of 2.0: the average fills more of its scale
        assert cr.bars[0][0][3] > tallest_max

    def test_the_scale_follows_the_series(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        chain, entries = namespace["_graph_chains"]()[0]
        graph = namespace["_ChainGraph"](chain, entries)
        assert graph.ymax() == 4.0                      # max displacement, 3.5
        namespace["GRAPH_SELECTED"][0] = 2              # CA/C1', highest is 1.0
        assert graph.ymax() == 1.0

    def test_a_residue_without_a_value_loses_its_bar(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        _graph, cr = drawn(namespace)
        assert len(cr.bars) == 3
        namespace["GRAPH_SELECTED"][0] = 2              # A 12 GLY has no CA/C1'
        _graph, cr = drawn(namespace)
        assert len(cr.bars) == 2

    def test_a_click_reports_the_shown_series(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        namespace["GRAPH_SELECTED"][0] = 1
        graph, _cr = drawn(namespace)
        status = types.SimpleNamespace(text="")
        status.set_text = lambda text: setattr(status, "text", text)
        pick = namespace["_make_pick"](graph, lambda x, y, z: None, status)
        start, end, _entry = graph.bars[0]
        pick((start + end) / 2.0)
        assert status.text == "A 10 SER    2.00 Å"

    def test_choosing_redraws_and_relabels(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        header = types.SimpleNamespace(text="")
        header.set_text = lambda text: setattr(header, "text", text)
        window = types.SimpleNamespace(destroyed=False)
        window.destroy = lambda: setattr(window, "destroyed", True)
        areas = fake_areas()
        choose = namespace["_make_choose"](2, window, areas, header)
        choose()
        assert namespace["_selected"]() == 2
        assert header.text == "CA/C1' displacement (Å)"
        assert [area.redraws for area, _size in areas] == [1, 1]
        assert window.destroyed

    def test_the_x_axis_does_not_move_with_the_series(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        graph, _cr = drawn(namespace)
        before = (graph.first, graph.last)
        namespace["GRAPH_SELECTED"][0] = 2
        graph, _cr = drawn(namespace)
        assert (graph.first, graph.last) == before


def apply_options(namespace, cutoffs=("0.5", "1", "2"), x_step="10",
                  y_step="1", ymax="", bar_width="7", max_width="30000"):
    """The Options window's Apply, with everything left at what it opens on."""
    return namespace["_apply_options"](list(cutoffs), x_step, y_step, ymax,
                                       bar_width, max_width)


class TestOptionsColours:
    """The colour bands, now a section of the Options window."""

    def test_new_cutoffs_are_applied(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert apply_options(namespace, cutoffs=("0.2", "0.4", "3")) == ""
        assert namespace["GRAPH_BANDS"] == [(0.2, "#55dd55"), (0.4, "#eecc22"),
                                            (3.0, "#ee9933"), (None, "#dd4444")]

    def test_the_colours_themselves_stay_put(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        before = [colour for _upper, colour in namespace["GRAPH_BANDS"]]
        apply_options(namespace, cutoffs=("1", "2", "3"))
        assert [colour for _upper, colour in namespace["GRAPH_BANDS"]] == before

    def test_bars_take_the_new_colours_on_the_next_draw(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        # 0.75 Å is yellow to start with, and orange once the bands tighten
        assert namespace["_band_colour"](0.75) == "#eecc22"
        apply_options(namespace, cutoffs=("0.2", "0.4", "3"))
        assert namespace["_band_colour"](0.75) == "#ee9933"
        _graph, cr = drawn(namespace)
        assert [colour for _rect, colour in cr.bars][2] == (0.933, 0.6, 0.2)

    def test_the_legend_follows_the_new_cutoffs(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        apply_options(namespace, cutoffs=("0.2", "0.4", "3"))
        cr = FakeCairo()
        namespace["_draw_legend"](cr, 560, 22)
        assert cr.texts == ["< 0.2", "0.2 - 0.4", "0.4 - 3", "> 3"]

    def test_a_cutoff_that_is_not_a_number_is_refused(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        problem = apply_options(namespace, cutoffs=("0.5", "wide", "2"))
        assert problem == "Cutoff: 'wide' is not a number."
        assert namespace["GRAPH_BANDS"] == list(COOT_GRAPH_BANDS)

    def test_cutoffs_have_to_increase(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        problem = apply_options(namespace, cutoffs=("2", "1", "3"))
        assert "larger than the one above" in problem
        assert namespace["GRAPH_BANDS"] == list(COOT_GRAPH_BANDS)

    def test_equal_cutoffs_are_refused(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert apply_options(namespace, cutoffs=("1", "1", "3")) != ""

    def test_cutoffs_have_to_be_positive(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        problem = apply_options(namespace, cutoffs=("0", "1", "2"))
        assert "greater than zero" in problem
        assert namespace["GRAPH_BANDS"] == list(COOT_GRAPH_BANDS)

    def test_surrounding_spaces_are_ignored(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert apply_options(namespace, cutoffs=(" 0.2 ", "0.4", "3 ")) == ""


class TestOptionsAxes:
    """The rest of the Options window: the two tick steps, the top, the bars."""

    def test_the_x_axis_step_is_taken(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert apply_options(namespace, x_step="20") == ""
        assert namespace["GRAPH_TICK_RESIDUES"] == [20]
        graph, cr = drawn(namespace)
        lowest = max(y for (_x, y), _text in cr.placed)
        assert [text for (_x, y), text in cr.placed if y == lowest] == ["20", "40"]

    def test_the_y_gridline_step_is_taken(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert apply_options(namespace, y_step="0.5") == ""
        _graph, cr = drawn(namespace)
        lowest = max(y for (_x, y), _text in cr.placed)
        labels = [text for (_x, y), text in cr.placed if y != lowest]
        assert labels == ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5"]

    def test_a_fixed_top_is_used_by_every_chain(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert apply_options(namespace, ymax="10") == ""
        chains = namespace["_graph_chains"]()
        scales = [namespace["_ChainGraph"](chain, entries).ymax()
                  for chain, entries in chains]
        assert scales == [10.0, 10.0]

    def test_a_blank_top_puts_each_chain_back_on_its_own(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        apply_options(namespace, ymax="10")
        apply_options(namespace, ymax="  ")
        chains = namespace["_graph_chains"]()
        scales = [namespace["_ChainGraph"](chain, entries).ymax()
                  for chain, entries in chains]
        assert scales == [4.0, 2.0]

    def test_a_bar_over_a_fixed_top_is_held_at_the_top(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        apply_options(namespace, ymax="1")             # chain A reaches 3.5
        _graph, cr = drawn(namespace)
        heights = [rect[3] for rect, _colour in cr.bars]
        assert max(heights) == namespace["GRAPH_HEIGHT"] - 10.0 - 26.0

    def test_the_bar_width_is_taken(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert apply_options(namespace, bar_width="14") == ""
        _graph, cr = drawn(namespace)
        assert [rect[2] for rect, _colour in cr.bars] == [14.0, 14.0, 14.0]

    def test_a_wider_bar_spreads_the_residues_out(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        graph, _cr = drawn(namespace)
        narrow = graph.width()
        apply_options(namespace, bar_width="14")
        graph, _cr = drawn(namespace)
        # the gap between bars is kept, so the pitch goes from 9 to 16
        assert graph.width() > narrow
        starts = [start for start, _end, _entry in graph.bars]
        assert round(starts[1] - starts[0], 6) == 2 * 16.0

    def long_chain(self, tmp_path, last=2904):
        """A chain the length of an rRNA, long enough to run out of canvas."""
        graph = [("A 1 G", "A", "1", (1.0,), "Å", 0.0, 0.0, 0.0),
                 ("A %d C" % last, "A", str(last), (2.0,), "Å", 1.0, 1.0, 1.0)]
        return graph_namespace(tmp_path, graph, graph_series=("Value (Å)",))

    def drawn_bar_width(self, namespace):
        """How wide the bars of that chain actually come out."""
        _graph, cr = drawn(namespace)
        return round(cr.bars[0][0][2], 2)

    def test_bar_width_still_bites_on_a_chain_that_has_to_be_squeezed(self, tmp_path):
        # the whole graph is capped, so a long chain cannot have the width it
        # asks for; it must still get wider bars for a wider setting
        namespace = self.long_chain(tmp_path)
        widths = []
        for asked in ("7", "14", "20"):
            apply_options(namespace, bar_width=asked)
            widths.append(self.drawn_bar_width(namespace))
        assert widths[0] < widths[1] < widths[2]

    def test_a_long_chain_is_no_longer_squeezed_at_the_default_width(self, tmp_path):
        namespace = self.long_chain(tmp_path)
        apply_options(namespace, bar_width="7")
        assert self.drawn_bar_width(namespace) == 7.0

    def test_the_graph_width_is_taken(self, tmp_path):
        namespace = self.long_chain(tmp_path)
        apply_options(namespace, bar_width="20", max_width="30000")
        squeezed = self.drawn_bar_width(namespace)
        apply_options(namespace, bar_width="20", max_width="10000")
        # less canvas for the same chain, so the bars have to give way further
        assert self.drawn_bar_width(namespace) < squeezed
        assert namespace["GRAPH_MAX_WIDTH"] == [10000]

    def test_the_graph_width_is_held_inside_what_gtk_can_draw(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        problem = apply_options(namespace, max_width="50000")
        assert "past what GTK can draw" in problem
        assert namespace["GRAPH_MAX_WIDTH"] == [30000]

    def test_a_useless_graph_width_is_refused(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert "or more" in apply_options(namespace, max_width="20")

    def test_a_bar_wider_than_the_limit_is_refused(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        problem = apply_options(namespace, bar_width="500")
        assert "or less" in problem
        assert namespace["GRAPH_BAR_WIDTH"] == [7.0]

    def test_the_x_step_has_to_be_whole_residues(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert "whole number" in apply_options(namespace, x_step="2.5")

    def test_each_field_is_checked(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        assert "Residues per label" in apply_options(namespace, x_step="none")
        assert "Gridline every" in apply_options(namespace, y_step="-1")
        assert "Highest y value" in apply_options(namespace, ymax="high")
        assert "Bar width" in apply_options(namespace, bar_width="0")

    def test_a_refused_edit_changes_nothing_at_all(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        apply_options(namespace, cutoffs=("0.2", "0.4", "3"), x_step="20",
                      y_step="0.5", ymax="9", bar_width="12")
        # one bad field, and the whole apply is dropped
        apply_options(namespace, cutoffs=("0.1", "0.2", "0.3"), x_step="5",
                      y_step="2", ymax="4", bar_width="wide")
        assert [upper for upper, _colour in namespace["GRAPH_BANDS"]] == [0.2, 0.4, 3.0, None]
        assert namespace["GRAPH_TICK_RESIDUES"] == [20]
        assert namespace["GRAPH_Y_STEP"] == [0.5]
        assert namespace["GRAPH_YMAX"] == [9.0]
        assert namespace["GRAPH_BAR_WIDTH"] == [12.0]

    def test_a_tiny_gridline_step_cannot_hang_the_draw(self, tmp_path):
        namespace = graph_namespace(tmp_path)
        apply_options(namespace, y_step="0.0001")
        _graph, cr = drawn(namespace)
        assert cr.gridlines <= namespace["GRAPH_Y_MAX_LINES"] + 1


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
