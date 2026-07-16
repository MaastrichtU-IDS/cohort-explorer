import asyncio
import importlib


def test_offline_sample_availability_reads_the_immutable_demo_pack(
    settings_factory,
    tmp_path,
    monkeypatch,
):
    pack = tmp_path / "pack"
    sample = pack / "dcr_output_TIME-CHF" / "shuffled_sample.csv"
    sample.parent.mkdir(parents=True)
    sample.write_text("age\n70\n", encoding="utf-8")
    settings = settings_factory(
        data_folder=str(tmp_path / "runtime"),
        demo_pack_dir=str(pack),
        offline_demo=True,
    )
    module = importlib.import_module("src.decentriq")
    monkeypatch.setattr(module, "settings", settings)

    result = asyncio.run(
        module.check_shuffled_samples(
            {"cohorts": {"TIME-CHF": {}, "GISSI-HF": {}}},
            user={"email": "nikolas.molyndris@decentriq.ch"},
        )
    )

    assert result == {
        "cohorts_with_samples": ["TIME-CHF"],
        "cohorts_without_samples": ["GISSI-HF"],
    }


def test_sample_availability_rejects_unsafe_cohort_ids(
    settings_factory,
    tmp_path,
    monkeypatch,
):
    settings = settings_factory(
        data_folder=str(tmp_path / "runtime"),
        demo_pack_dir=str(tmp_path / "pack"),
        offline_demo=True,
    )
    module = importlib.import_module("src.decentriq")
    monkeypatch.setattr(module, "settings", settings)

    result = asyncio.run(
        module.check_shuffled_samples(
            {"cohorts": {"../../outside": {}}},
            user={"email": "nikolas.molyndris@decentriq.ch"},
        )
    )

    assert result == {
        "cohorts_with_samples": [],
        "cohorts_without_samples": ["../../outside"],
    }
