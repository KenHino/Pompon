import pytest


def test_import() -> None:
    import pompon

    print(pompon.__version__)


if __name__ == "__main__":
    pytest.main([__file__])
