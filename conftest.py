import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--repaired-path",
        action="store",
        default=None,
        help="Path to repaired function file to test instead of buggy_function.py",
    )


@pytest.fixture
def repaired_path(request):
    return request.config.getoption("--repaired-path")
