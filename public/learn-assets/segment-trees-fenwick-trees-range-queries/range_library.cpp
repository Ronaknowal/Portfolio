// C++17. Official AC Library v1.6 headers; see the lesson's setup command.
#include <atcoder/segtree>
#include <atcoder/fenwicktree>
#include <atcoder/lazysegtree>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using Integer = long long;
Integer sum(Integer first, Integer second) { return first + second; }
Integer zero() { return 0; }
std::string join(std::string first, std::string second) { return first + second; }
std::string empty_text() { return ""; }

struct Summary { Integer total, length; };
// A tag maps each scalar x to multiplier*x + offset.
// Addition d is {1,d}; assignment c is {0,c}.
struct Action { Integer multiplier, offset; };
Summary combine(Summary first, Summary second) {
    return {first.total + second.total, first.length + second.length};
}
Summary identity() { return {0, 0}; }
Summary apply(Action action, Summary node) {
    return {action.multiplier * node.total + action.offset * node.length, node.length};
}
Action compose(Action newer, Action older) {
    return {newer.multiplier * older.multiplier,
            newer.multiplier * older.offset + newer.offset};
}
Action unchanged() { return {1, 0}; }

int main() {
    std::vector<Integer> values = {2, 1, 3, 4, 0, 5, 2, 1};
    atcoder::segtree<Integer, sum, zero> segment(values);
    atcoder::fenwick_tree<Integer> fenwick(values.size());
    for (int i = 0; i < int(values.size()); ++i) fenwick.add(i, values[i]);
    assert(segment.prod(1, 7) == 15 && fenwick.sum(1, 7) == 15);
    segment.set(4, 3);
    fenwick.add(4, 3 - values[4]);
    values[4] = 3;
    assert(segment.prod(1, 7) == 18 && fenwick.sum(1, 7) == 18);
    std::cout << "sum before / after: 15 18\n";
    atcoder::segtree<std::string, join, empty_text> ordered(
        std::vector<std::string>{"A", "B", "C", "D"});
    assert(ordered.prod(1, 4) == "BCD" && ordered.prod(2, 2).empty());
    std::cout << "ordered fold: " << ordered.prod(1, 4) << '\n';
    std::vector<Summary> initial;
    for (Integer value : std::vector<Integer>{2, 1, 3, 4, 0}) initial.push_back({value, 1});
    atcoder::lazy_segtree<Summary, combine, identity, Action, apply, compose, unchanged> lazy(initial);
    lazy.apply(1, 4, {1, 3});   // [2,4,6,7,0]
    lazy.apply(2, 5, {0, -2});  // [2,4,-2,-2,-2]
    lazy.apply(3, 5, {1, 5});   // [2,4,-2,3,3]
    assert(lazy.prod(0, 5).total == 10);
    assert(lazy.prod(2, 4).total == 1);
    std::cout << "lazy total / [2,4): " << lazy.prod(0, 5).total << ' '
              << lazy.prod(2, 4).total << '\n';
    // Explicit order check: add after set is different from set after add.
    assert(apply(compose({1, 5}, {0, -2}), {9, 1}).total == 3);
    assert(apply(compose({0, -2}, {1, 5}), {9, 1}).total == -2);
    atcoder::segtree<Integer, sum, zero> no_values(0);
    assert(no_values.prod(0, 0) == 0);
    std::cout << "empty fold: " << no_values.prod(0, 0) << '\n';
}
