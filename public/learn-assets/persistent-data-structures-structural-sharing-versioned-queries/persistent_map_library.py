"""Run beside persistent_structures_mechanisms.py; pip install immutables==0.21."""
from immutables import Map
from persistent_structures_mechanisms import build, assign, range_sum


def main():
    values = [2, 1, 4, 3, 5]
    roots = [build(values)]
    maps = [Map(enumerate(values))]
    # Each edit names a parent version; later edits can branch from old parents.
    for parent, index, value in [(0, 2, 9), (0, 4, -2), (1, 0, 8), (1, 2, 9)]:
        roots.append(assign(roots[parent], index, value))
        maps.append(maps[parent].set(index, value))
    for version, (root, mapping) in enumerate(zip(roots, maps)):
        actual = [mapping[i] for i in range(len(values))]
        assert all(range_sum(root, i, i + 1) == actual[i] for i in range(len(values)))
        assert range_sum(root, 1, 5) == sum(mapping[i] for i in range(1, 5))
        print('version', version, actual)
    assert roots[4] is roots[1]  # A property of our tree, not a Map identity promise.
    with maps[0].mutate() as draft:
        draft[0] = 20
        draft[1] = 10
        edited = draft.finish()
    print('batch / unchanged original:', [edited[0], edited[1]], [maps[0][0], maps[0][1]])
    # Immutability protects the mapping structure, not a mutable payload object.
    payload = []
    shallow = Map({'records': payload})
    payload.append(7)
    print('mutable payload remains shared:', shallow['records'])


if __name__ == '__main__':
    main()
