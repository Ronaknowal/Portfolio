"""Run beside randomized_algorithm_mechanisms.py. Standard library only."""
from collections import Counter
from random import Random
from randomized_algorithm_mechanisms import shuffled, reservoir, WeightedPicker


def main():
    records = list(enumerate(['same', 'other', 'same', 'last']))
    scratch_shuffle = shuffled(records, Random(17))
    library_shuffle = records.copy()
    Random(17).shuffle(library_shuffle)
    assert Counter(scratch_shuffle) == Counter(library_shuffle) == Counter(records)
    print('shuffle preserves occurrences:', True)
    scratch_sample = reservoir(iter(records), 2, Random(17))
    library_sample = Random(17).sample(records, k=2)
    assert len({i for i, _ in scratch_sample}) == len({i for i, _ in library_sample}) == 2
    print('reservoir / sample:', scratch_sample, library_sample)
    print('short stream:', reservoir(iter(records), 9, Random(17)))
    try:
        Random(17).sample(records, k=9)
    except ValueError:
        print('sample larger than population:', 'ValueError')

    weights = [0, 2, 0, 1]
    picker = WeightedPicker(weights)
    scratch_rng, library_rng = Random(17), Random(17)
    exact_tickets = [picker.pick(scratch_rng) for _ in range(8)]
    float_tickets = library_rng.choices(range(4), weights=weights, k=8)
    assert set(exact_tickets + float_tickets) <= {1, 3}
    print('integer tickets / choices:', exact_tickets, float_tickets)
    # Save and restore the state of the SAME API; equal seeds across algorithms
    # are not a contract for identical samples or identical random consumption.
    checkpoint = library_rng.getstate()
    first = library_rng.sample(records, k=3)
    library_rng.setstate(checkpoint)
    assert first == library_rng.sample(records, k=3)
    print('restored state replays:', True)


if __name__ == '__main__':
    main()
