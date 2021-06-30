#include "ppl/nn/runtime/sequence.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;

struct Value {
    Value() {}
    Value(int vv) : v(vv) {}
    int v;
};

TEST(SequenceTest, misc) {
    vector<Value> values = {
        363, 521, 16556, 5345, 974,
    };

    Sequence<Value> seq(nullptr);
    for (auto x = values.begin(); x != values.end(); ++x) {
        seq.EmplaceBack(Value(x->v));
    }

    EXPECT_EQ(values.size(), seq.GetElementCount());
    for (uint32_t i = 0; i < seq.GetElementCount(); ++i) {
        auto element = seq.GetElement(i);
        EXPECT_NE(nullptr, element);
        EXPECT_EQ(values[i].v, element->v);
    }
}
