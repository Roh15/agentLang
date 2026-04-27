"""
Quality oracle for the LRUCache experiment.

Loads LRUCache from --repaired-path if given, otherwise from buggy_lru_cache.py.
Run against the repaired output:
    pytest test_buggy_lru_cache.py --repaired-path=repaired_lru_cache.py -v
"""
import importlib.util
import sys
import pytest


def load_lru_cache_class(path):
    spec = importlib.util.spec_from_file_location("lru_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LRUCache


@pytest.fixture(scope="module")
def LRUCache(request):
    path = request.config.getoption("--repaired-path") or "buggy_lru_cache.py"
    return load_lru_cache_class(path)


# ── Basic get / put ────────────────────────────────────────────────────────────

def test_get_returns_minus_one_when_absent(LRUCache):
    c = LRUCache(2)
    assert c.get(1) == -1


def test_put_and_get_single_item(LRUCache):
    c = LRUCache(2)
    c.put(1, 10)
    assert c.get(1) == 10


def test_put_overwrites_existing_key(LRUCache):
    c = LRUCache(2)
    c.put(1, 10)
    c.put(1, 99)
    assert c.get(1) == 99


def test_len_tracks_insertions(LRUCache):
    c = LRUCache(3)
    assert len(c) == 0
    c.put(1, 1)
    assert len(c) == 1
    c.put(2, 2)
    assert len(c) == 2


def test_contains(LRUCache):
    c = LRUCache(2)
    c.put(5, 50)
    assert 5 in c
    assert 9 not in c


# ── Capacity enforcement ───────────────────────────────────────────────────────

def test_capacity_one_evicts_on_second_put(LRUCache):
    c = LRUCache(1)
    c.put(1, 10)
    c.put(2, 20)
    assert c.get(2) == 20
    assert c.get(1) == -1


def test_capacity_respected_exactly(LRUCache):
    """Cache holds exactly `capacity` items, not capacity-1."""
    c = LRUCache(3)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    # All three should still be present — no eviction yet
    assert c.get(1) == 1
    assert c.get(2) == 2
    assert c.get(3) == 3
    assert len(c) == 3


def test_fourth_put_evicts_lru(LRUCache):
    c = LRUCache(3)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    c.put(4, 4)       # should evict key 1 (LRU)
    assert c.get(1) == -1
    assert c.get(4) == 4
    assert len(c) == 3


# ── LRU eviction order ────────────────────────────────────────────────────────

def test_evicts_least_recently_used_not_oldest(LRUCache):
    """get() updates recency — key 1 accessed last so key 2 is LRU."""
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.get(1)          # key 1 now MRU; key 2 is LRU
    c.put(3, 3)       # evicts key 2
    assert c.get(2) == -1
    assert c.get(1) == 1
    assert c.get(3) == 3


def test_put_existing_key_updates_recency(LRUCache):
    """put() on existing key promotes it to MRU so it is not evicted."""
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(1, 100)     # key 1 refreshed → key 2 is now LRU
    c.put(3, 3)       # evicts key 2
    assert c.get(2) == -1
    assert c.get(1) == 100
    assert c.get(3) == 3


def test_evicts_correct_item_after_multiple_accesses(LRUCache):
    c = LRUCache(3)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    c.get(1)          # order: 1 (MRU), 3, 2 (LRU)
    c.get(3)          # order: 3 (MRU), 1, 2 (LRU)
    c.put(4, 4)       # evicts key 2
    assert c.get(2) == -1
    assert c.get(1) == 1
    assert c.get(3) == 3
    assert c.get(4) == 4


# ── peek doesn't affect recency ────────────────────────────────────────────────

def test_peek_does_not_update_recency(LRUCache):
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.peek(1)         # key 1 peeked — recency must NOT change; key 1 still LRU
    c.put(3, 3)       # should evict key 1
    assert c.get(1) == -1
    assert c.get(2) == 2
    assert c.get(3) == 3


def test_peek_returns_correct_value(LRUCache):
    c = LRUCache(2)
    c.put(7, 77)
    assert c.peek(7) == 77
    assert c.peek(99) == -1


# ── get_all_keys ordering ──────────────────────────────────────────────────────

def test_get_all_keys_mru_to_lru_order(LRUCache):
    c = LRUCache(3)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    # Insertion order: 1, 2, 3 → MRU=3, LRU=1
    assert c.get_all_keys() == [3, 2, 1]


def test_get_all_keys_reflects_access(LRUCache):
    c = LRUCache(3)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)
    c.get(1)          # key 1 now MRU → order: 1, 3, 2
    assert c.get_all_keys() == [1, 3, 2]


def test_get_all_keys_empty_cache(LRUCache):
    c = LRUCache(3)
    assert c.get_all_keys() == []
