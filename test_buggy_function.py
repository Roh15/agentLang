import importlib.util
import sys
import pytest


def load_binary_search(path):
    spec = importlib.util.spec_from_file_location("target_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.binary_search


@pytest.fixture(scope="module")
def binary_search(request):
    path = request.config.getoption("--repaired-path") or "buggy_function.py"
    return load_binary_search(path)


def test_found_first(binary_search):
    assert binary_search([1, 2, 3, 4, 5], 1) == 0


def test_found_last(binary_search):
    assert binary_search([1, 2, 3, 4, 5], 5) == 4


def test_found_middle(binary_search):
    assert binary_search([1, 2, 3, 4, 5], 3) == 2


def test_not_found_returns_minus_one(binary_search):
    assert binary_search([1, 2, 3, 4, 5], 6) == -1


def test_does_not_mutate_input(binary_search):
    arr = [3, 1, 4, 1, 5]
    original = arr[:]
    binary_search(arr, 4)
    assert arr == original


def test_empty_list(binary_search):
    assert binary_search([], 99) == -1
