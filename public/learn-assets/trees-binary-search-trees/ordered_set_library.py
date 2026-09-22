"""Run beside tree_mechanisms.py. Python 3.12+, standard library only."""
from bisect import bisect_left, bisect_right
from tree_mechanisms import build, insert, delete, inorder, floor_key, ceiling_key


def add_unique(ordered, key):
    position = bisect_left(ordered, key)
    if position == len(ordered) or ordered[position] != key:
        ordered.insert(position, key)


def discard(ordered, key):
    position = bisect_left(ordered, key)
    if position < len(ordered) and ordered[position] == key:
        ordered.pop(position)


def neighbors(ordered, target):
    below = bisect_right(ordered, target) - 1
    above = bisect_left(ordered, target)
    return (ordered[below] if below >= 0 else None,
            ordered[above] if above < len(ordered) else None)


def main():
    keys = [8, 3, 10, 1, 6, 14, 4, 7, 13]
    root, ordered = build(keys), sorted(set(keys))
    for operation, key in [('add', 6), ('add', 5), ('remove', 8), ('remove', 99)]:
        if operation == 'add':
            root = insert(root, key)
            add_unique(ordered, key)
        else:
            root = delete(root, key)
            discard(ordered, key)
        assert inorder(root) == ordered
    print('ordered:', ordered)
    for target in [0, 6, 8, 20]:
        actual = neighbors(ordered, target)
        assert actual == (floor_key(root, target), ceiling_key(root, target))
        print('floor / ceiling', target, actual)
    print('inclusive [4, 10]:', ordered[bisect_left(ordered, 4):bisect_right(ordered, 10)])
    print('empty:', neighbors([], 4))


if __name__ == '__main__':
    main()
