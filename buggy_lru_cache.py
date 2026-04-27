class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    """
    LRU cache backed by a doubly-linked list (recency order) and a dict (O(1) lookup).

    Sentinel head = most-recently-used end.
    Sentinel tail = least-recently-used end.

    Operations: get(key) -> val | -1, put(key, val), peek(key) -> val | -1,
                get_all_keys() -> list in MRU-to-LRU order.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.cache = {}          # key -> Node
        self.head = Node(0, 0)   # sentinel MRU end
        self.tail = Node(0, 0)   # sentinel LRU end
        self.head.next = self.tail
        # Bug 1: self.tail.prev = self.head is missing.
        # self.tail.prev is None; the first eviction calls self.tail.prev.next
        # and raises AttributeError: 'NoneType' object has no attribute 'next'.

    def _remove(self, node):
        """Unlink node from the doubly-linked list."""
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = node          # Bug 2: should be nxt.prev = prev.
                                 # Sets nxt's backward pointer to the node just removed,
                                 # so traversing backward from nxt leads back to a stale node.

    def _insert_front(self, node):
        """Insert node immediately after sentinel head (most-recently-used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next = node           # Bug 3: this line and the next are swapped.
        self.head.next.prev = node      # head.next is now node itself, so this sets
                                        # node.prev = node — a self-referential prev pointer.
                                        # Correct order: update old head.next's prev first,
                                        # then redirect head.next to node.

    def get(self, key):
        """Return value for key and mark it most-recently-used. Returns -1 if absent."""
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # Bug 4: recency is never updated. The two lines below are missing:
        #   self._remove(node)
        #   self._insert_front(node)
        # Every get() leaves the node in its current list position, so LRU
        # order is never updated and the wrong item gets evicted.
        return node.val

    def put(self, key, val):
        """Insert or update key=val. Evicts LRU entry when over capacity."""
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, val)
        self.cache[key] = node
        self._insert_front(node)
        if len(self.cache) >= self.capacity:   # Bug 5: >= should be >.
                                               # Evicts when AT capacity rather than
                                               # when OVER it — effective capacity is
                                               # capacity-1, one slot short.
            lru = self.head.next               # Bug 6: should be self.tail.prev.
                                               # head.next is the MRU node, not the LRU.
                                               # This evicts the item just inserted,
                                               # making every put a no-op when full.
            self._remove(lru)
            del self.cache[lru.key]

    def peek(self, key):
        """Return value for key without updating recency. Returns -1 if absent."""
        if key not in self.cache:
            return -1
        return self.cache[key].val

    def get_all_keys(self):
        """Return all keys in most-recently-used to least-recently-used order."""
        keys = []
        node = self.tail.prev              # Bug 7: should be self.head.next.
        while node != self.head:           # Starts at the LRU end and walks backward
            keys.append(node.key)          # via node.prev, yielding LRU-to-MRU order
            node = node.prev               # (the reverse of what the docstring promises).
        return keys

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache
