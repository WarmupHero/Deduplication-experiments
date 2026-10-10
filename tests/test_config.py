from dedup_bench import config


def test_project_root_is_the_repository_root() -> None:
    assert (config.PROJ_ROOT / "pyproject.toml").is_file()


def test_data_directories_sit_under_the_data_directory() -> None:
    data_dirs = (
        config.RAW_DATA_DIR,
        config.INTERIM_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.EXTERNAL_DATA_DIR,
    )
    assert all(path.parent == config.DATA_DIR for path in data_dirs)
