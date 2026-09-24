// Reuse the actual lesson's adapter functions, compare every result to arrays.
#define main recorded_lesson_main
#include "range_library.cpp"
#undef main
#include <numeric>
#include <random>

int main() {
    std::mt19937 rng(221);
    int checked = 0;
    for (int size : {0, 1, 3, 5, 13}) {
        std::vector<Integer> values(size);
        for (auto& value : values) value = Integer(rng() % 19) - 9;
        atcoder::segtree<Integer, sum, zero> point(values);
        atcoder::fenwick_tree<Integer> fenwick(size);
        std::vector<Summary> initial;
        for (int i = 0; i < size; ++i) {
            fenwick.add(i, values[i]);
            initial.push_back({values[i], 1});
        }
        atcoder::lazy_segtree<Summary, combine, identity, Action, apply, compose, unchanged> lazy(initial);
        for (int step = 0; step < 100; ++step) {
            int left = rng() % (size + 1), right = rng() % (size + 1);
            if (left > right) std::swap(left, right);
            Integer value = Integer(rng() % 17) - 8;
            Action action = (step % 2) ? Action{0, value} : Action{1, value};
            lazy.apply(left, right, action);
            for (int i = left; i < right; ++i) {
                Integer next = action.multiplier * values[i] + action.offset;
                fenwick.add(i, next - values[i]);
                point.set(i, next);
                values[i] = next;
            }
            for (int low = 0; low <= size; ++low) {
                for (int high = low; high <= size; ++high) {
                    Integer expected = std::accumulate(values.begin() + low, values.begin() + high, Integer(0));
                    assert(point.prod(low, high) == expected);
                    assert(fenwick.sum(low, high) == expected);
                    assert(lazy.prod(low, high).total == expected);
                    ++checked;
                }
            }
        }
    }
    std::vector<std::string> letters = {"q", "ab", "", "Z", "w"};
    atcoder::segtree<std::string, join, empty_text> ordered(letters);
    for (int low = 0; low <= 5; ++low) {
        for (int high = low; high <= 5; ++high) {
            std::string expected;
            for (int i = low; i < high; ++i) expected += letters[i];
            assert(ordered.prod(low, high) == expected);
        }
    }
    std::cout << "C++ direct-array range checks: " << checked << "; ordered folds: 21\n";
}
