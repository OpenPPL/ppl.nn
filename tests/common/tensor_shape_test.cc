#include "ppl/nn/common/tensor_shape.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::common;

TEST(TensorShapeTest, type_and_format) {
    TensorShape s;
    s.SetDataType(DATATYPE_FLOAT32);
    ASSERT_EQ(DATATYPE_FLOAT32, s.GetDataType());
    s.SetDataFormat(DATAFORMAT_NDARRAY);
    ASSERT_EQ(DATAFORMAT_NDARRAY, s.GetDataFormat());
}

TEST(TensorShapeTest, dim_and_padding) {
    TensorShape s;

    s.SetDimCount(4);
    ASSERT_EQ(4, s.GetDimCount());

    s.SetDim(0, 1);
    s.SetDim(1, 100);
    s.SetDim(2, 1024);
    s.SetDim(3, 2048);
    ASSERT_EQ(1, s.GetDim(0));
    ASSERT_EQ(100, s.GetDim(1));
    ASSERT_EQ(1024, s.GetDim(2));
    ASSERT_EQ(2048, s.GetDim(3));

    s.SetPadding0(1, 2);
    s.SetPadding1(1, 4);
    ASSERT_EQ(2, s.GetPadding0(1));
    ASSERT_EQ(4, s.GetPadding1(1));
}

TEST(TensorShapeTest, elements_and_bytes) {
    TensorShape s;
    int64_t dims[] = {1, 100, 1024, 2048};
    s.Reshape(dims, 4);

    s.SetDataType(DATATYPE_FLOAT32);
    s.SetDataFormat(DATAFORMAT_NDARRAY);

    ASSERT_EQ(209715200, s.GetElementsIncludingPadding());
    ASSERT_EQ(209715200, s.GetElementsExcludingPadding());
    ASSERT_EQ(838860800, s.GetBytesIncludingPadding());
    ASSERT_EQ(838860800, s.GetBytesExcludingPadding());

    s.SetPadding0(1, 2);
    s.SetPadding1(1, 2);

    ASSERT_EQ(218103808, s.GetElementsIncludingPadding());
    ASSERT_EQ(209715200, s.GetElementsExcludingPadding());
    ASSERT_EQ(872415232, s.GetBytesIncludingPadding());
    ASSERT_EQ(838860800, s.GetBytesExcludingPadding());
}
