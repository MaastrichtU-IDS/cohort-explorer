"""Regression tests for the cell-state model used by the EDA compute nodes.

Every cell of a variable is classified into exactly one of:
    valid | empty | coded_missing | invalid
and those four counts must always sum to the number of rows.

The helpers live inside the ``_SHARED_HELPERS`` source block that gets injected
into the c2 and c3 Decentriq scripts, so they are exec'd here to be tested.

Run with:  uv run pytest backend/test_missing_model.py -q
"""

import os
import sys
import tempfile
import zipfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import eda_scripts  # noqa: E402

_ns = {"pd": pd, "os": os, "zipfile": zipfile, "tempfile": tempfile}
exec(eda_scripts._SHARED_HELPERS, _ns)

classify_series = _ns["classify_series"]
candidate_tokens = _ns["candidate_tokens"]
_parse_missing_codes = _ns["_parse_missing_codes"]
_norm_token = _ns["_norm_token"]
load_data_as_text = _ns["load_data_as_text"]


def test_missing_codes_may_be_pipe_separated():
    assert _parse_missing_codes("999|-1") == ["999", "-1"]
    assert _parse_missing_codes(" 999 | -1 ") == ["999", "-1"]
    assert _parse_missing_codes("999") == ["999"]
    assert _parse_missing_codes("") == []
    assert _parse_missing_codes(float("nan")) == []
    assert _parse_missing_codes(999) == ["999"]


def test_code_matches_even_when_written_as_a_float():
    # A float column renders the code 999 as "999.0"; it must still match the
    # "999" declared in the dictionary.
    assert _norm_token("999.0") == _norm_token("999")
    assert _norm_token("-1.00") == _norm_token("-1")
    assert _norm_token(999.0) == "999"


def test_column_with_both_coded_missing_and_empty_cells():
    col = pd.Series(
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
        + ["", "  ", ""]
        + ["999", "999", "-1", "-1"]
        + ["abc"]
    )
    comp, valid = classify_series(col, ["999", "-1"], "int")

    assert comp["n_rows"] == 20
    assert comp["n_valid"] == 12
    assert comp["n_empty"] == 3
    assert comp["n_coded_missing"] == 4
    assert comp["n_invalid"] == 1
    assert comp["n_missing_total"] == 7
    assert comp["coded_missing_by_code"] == {"999": 2, "-1": 2}
    assert comp["pct_missing_total"] == 35.0
    assert comp["partition_check_ok"] is True
    assert sorted(int(x) for x in valid) == list(range(1, 13))


def test_float_column_codes_are_not_averaged_in():
    col = pd.Series(["1.5", "2.5", "", "999.0", "-1.0", "999.0"])
    comp, valid = classify_series(col, ["999", "-1"], "float")

    assert comp["n_valid"] == 2
    assert comp["n_empty"] == 1
    assert comp["n_coded_missing"] == 3
    assert comp["coded_missing_by_code"] == {"999": 2, "-1": 1}
    assert list(valid) == [1.5, 2.5]


def test_categorical_code_is_not_a_category():
    comp, valid = classify_series(pd.Series(["1", "1", "2", "", "9", "9", "2"]), ["9"], "categorical")

    assert comp["n_valid"] == 4
    assert comp["n_empty"] == 1
    assert comp["n_coded_missing"] == 2
    assert comp["partition_check_ok"] is True
    assert sorted(set(valid)) == ["1", "2"]


def test_categorical_codes_written_as_floats_match_dictionary_keys():
    # Exporters often write categorical codes as "1.0"; the dictionary declares
    # them as "1", so the label lookup must still succeed.
    comp, valid = classify_series(pd.Series(["1.0", "2.0", "1.0", ""]), [], "categorical")

    assert sorted(set(valid)) == ["1", "2"]
    assert comp["n_valid"] == 3
    assert comp["n_empty"] == 1

    # text categories keep their original form
    _comp, valid_txt = classify_series(pd.Series(["Male", "Female"]), [], "categorical")
    assert sorted(set(valid_txt)) == ["Female", "Male"]


def test_dates_split_into_all_four_states():
    col = pd.Series(["2020-01-01", "", "1900-01-01", "2021-05-05", "not-a-date"])
    comp, _ = classify_series(col, ["1900-01-01"], "date")

    assert comp["n_valid"] == 2
    assert comp["n_empty"] == 1
    assert comp["n_coded_missing"] == 1
    assert comp["n_invalid"] == 1
    assert comp["partition_check_ok"] is True


def test_no_codes_declared_means_blanks_are_only_empty():
    comp, _ = classify_series(pd.Series(["1", "", "2"]), [], "int")

    assert comp["n_coded_missing"] == 0
    assert comp["n_empty"] == 1
    assert comp["partition_check_ok"] is True


def test_type_inference_ignores_blanks_and_codes():
    cands = candidate_tokens(pd.Series(["1", "2", "3", "999", ""]), ["999"])
    assert sorted(cands.tolist()) == ["1", "2", "3"]


def test_loader_keeps_na_literal_and_blanks_blank():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "d.csv")
        with open(path, "w") as fh:
            fh.write("a,b\n1,NA\n,2\n999,nan\n")

        df, _notes = load_data_as_text(path)

        assert df["a"][1] == ""
        assert df["b"][0] == "NA"
        assert df["b"][2] == "nan"
        assert int(df.isna().sum().sum()) == 0

        comp, _ = classify_series(df["b"], [], "categorical")
        assert comp["n_valid"] == 3


def _zip_of(td, files):
    paths = []
    for name, body in files.items():
        p = os.path.join(td, name)
        with open(p, "w") as fh:
            fh.write(body)
        paths.append((p, name))
    z = os.path.join(td, "bundle.zip")
    with zipfile.ZipFile(z, "w") as zf:
        for p, name in paths:
            zf.write(p, name)
    return z


def test_zip_with_mismatched_headers_is_refused():
    with tempfile.TemporaryDirectory() as td:
        z = _zip_of(td, {"a.csv": "x,y\n1,2\n", "b.csv": "x,z\n3,4\n"})
        try:
            load_data_as_text(z)
        except ValueError as exc:
            assert "Header mismatch" in str(exc)
        else:
            raise AssertionError("expected a header mismatch to be refused")


def test_zip_with_matching_headers_concatenates():
    with tempfile.TemporaryDirectory() as td:
        z = _zip_of(td, {"a.csv": "x,y\n1,2\n", "b.csv": "x,y\n3,4\n"})
        df, _notes = load_data_as_text(z)
        assert len(df) == 2
