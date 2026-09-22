// Independent algebra checks against the learner-visible AtCoder adapter.
#define main recorded_lesson_main
#include "range_library.cpp"
#undef main

int main() {
    int cases = 0;
    for (Integer a : {-2, -1, 0, 1, 2}) {
        for (Integer b : {-3, 0, 4}) {
            for (Integer c : {-1, 0, 2}) {
                for (Integer d : {-4, 0, 3}) {
                    Action first{a,b}, second{c,d};
                    for (Integer x : {-5, 0, 7}) {
                        for (Integer y : {-2, 1, 6}) {
                            Summary left{x,1}, right{y,1};
                            auto nested = apply(first, apply(second, combine(left,right)));
                            auto composed = apply(compose(first,second), combine(left,right));
                            auto distributed = combine(apply(first,left),apply(first,right));
                            assert(nested.total == composed.total && nested.length == composed.length);
                            assert(apply(first,combine(left,right)).total == distributed.total);
                            assert(apply(first,identity()).total == 0 && apply(first,identity()).length == 0);
                            ++cases;
                        }
                    }
                }
            }
        }
    }
    std::vector<Summary> initial{{2,1},{-1,1},{4,1}};
    atcoder::lazy_segtree<Summary,combine,identity,Action,apply,compose,unchanged> tree(initial);
    tree.apply(0,3,{-1,2});  // [0,3,-2]
    tree.apply(1,3,{2,-1}); // [0,5,-5]
    assert(tree.prod(0,3).total == 0 && tree.prod(1,2).total == 5 && tree.prod(2,3).total == -5);
    std::cout << "independent affine-action laws: " << cases << "; composed negative-multiplier path passed\n";
}
