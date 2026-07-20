import pytest


def test_offline_demo_rejects_live_metadata_provider(settings_factory):
    settings = settings_factory(
        dev_mode=True,
        offline_demo=True,
        concept_search_backend="athena",
        concept_validation_backend="fixture",
        mapping_generation_backend="fixture",
    )

    with pytest.raises(ValueError, match="fixture metadata providers"):
        settings.validate_runtime()


def test_local_auth_requires_development_mode(settings_factory):
    settings = settings_factory(dev_mode=False, local_auth_enabled=True)

    with pytest.raises(ValueError, match="LOCAL_AUTH_ENABLED requires DEV_MODE=true"):
        settings.validate_runtime()


def test_runtime_requires_explicit_jwt_secret(settings_factory):
    settings = settings_factory(jwt_secret="")

    with pytest.raises(ValueError, match="JWT_SECRET is required"):
        settings.validate_runtime()


def test_aadcr_runtime_rejects_unknown_handoff_mode(settings_factory):
    settings = settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_jwt_secret="test-secret",  # noqa: S106 - synthetic test value
        aadcrv2_handoff_mode="automatic-ish",
    )

    with pytest.raises(ValueError, match="AADCRV2_HANDOFF_MODE"):
        settings.validate_runtime()


def test_guarded_local_admin_is_added_to_admins(settings_factory):
    settings = settings_factory(
        admins="existing@example.test",
        dev_mode=True,
        local_auth_email="nikolas.molyndris@decentriq.ch",
        local_auth_enabled=True,
    )

    assert settings.admins_list == [
        "existing@example.test",
        "nikolas.molyndris@decentriq.ch",
    ]


def test_disabled_local_admin_is_not_added_to_admins(settings_factory):
    settings = settings_factory(
        admins="existing@example.test",
        dev_mode=True,
        local_auth_email="nikolas.molyndris@decentriq.ch",
        local_auth_enabled=False,
    )

    assert settings.admins_list == ["existing@example.test"]


def test_application_startup_validates_runtime(settings_factory, import_main_with_stubs):
    settings = settings_factory(jwt_secret="")

    with pytest.raises(ValueError, match="JWT_SECRET is required"):
        import_main_with_stubs(settings)
