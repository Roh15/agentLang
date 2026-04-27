def binary_search(arr, target):
    """
    Search for target in a sorted list. Returns the index of target, or -1 if not found.
    """
    arr.sort()  # Bug 1: mutates caller's list in place

    low = 0
    high = len(arr)  # Bug 2: should be len(arr) - 1

    while low < high:  # Bug 3: should be low <= high (misses the case low == high)
        mid = low + high // 2  # Bug 4: wrong midpoint — should be (low + high) // 2

        if arr[mid] == target:
            return mid + 1  # Bug 5: off-by-one, should return mid

        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    # Bug 6: falls off the end returning None instead of -1
