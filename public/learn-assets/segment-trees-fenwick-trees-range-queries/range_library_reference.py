"""Run beside range_query_mechanisms.py; compare with range_library.cpp output."""
from range_query_mechanisms import SegmentTree, Fenwick, LazySum


def main():
    values = [2, 1, 3, 4, 0, 5, 2, 1]
    segment = SegmentTree(values, lambda a, b: a + b, 0)
    fenwick = Fenwick(values)
    assert segment.query(1, 7) == fenwick.range_sum(1, 7) == 15
    segment.set(4, 3)
    fenwick.set(4, 3)
    assert segment.query(1, 7) == fenwick.range_sum(1, 7) == 18
    print('sum before / after: 15 18')
    ordered = SegmentTree(list('ABCD'), lambda a, b: a + b, '')
    assert ordered.query(1, 4) == 'BCD' and ordered.query(2, 2) == ''
    print('ordered fold:', ordered.query(1, 4))
    lazy = LazySum([2, 1, 3, 4, 0])
    lazy.add(1, 4, 3)
    lazy.set(2, 5, -2)
    lazy.add(3, 5, 5)
    assert lazy.query(0, 5) == 10 and lazy.query(2, 4) == 1
    print('lazy total / [2,4):', lazy.query(0, 5), lazy.query(2, 4))
    print('empty fold:', SegmentTree([], lambda a, b: a + b, 0).query(0, 0))


if __name__ == '__main__':
    main()
