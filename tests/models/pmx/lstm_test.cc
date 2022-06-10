#include "pmx_utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/lstm.h"

using namespace std;
using namespace ppl::nn::onnx;
using namespace ppl::nn::pmx::onnx;

TEST_F(PmxTest, test_lstm) {
    DEFINE_ARG(LSTMParam, lstm);
    lstm_param1.activation_alpha = {0.23f};
    lstm_param1.activation_beta = {0.33f};
    lstm_param1.activations = {ppl::nn::onnx::LSTMParam::ACT_ELU};
    lstm_param1.clip = 0.34;
    lstm_param1.direction = ppl::nn::onnx::LSTMParam::DIR_REVERSE;
    lstm_param1.hidden_size = 44;
    lstm_param1.input_forget = 23;
    MAKE_BUFFER(LSTMParam, lstm);
    std::vector<float> activation_alpha = lstm_param3.activation_alpha;
    std::vector<float> activation_beta = lstm_param3.activation_beta;
    std::vector<int32_t> activations = lstm_param3.activations;
    float clip = lstm_param3.clip;
    int32_t direction = lstm_param3.direction;
    int32_t hidden_size = lstm_param3.hidden_size;
    int32_t input_forget = lstm_param3.input_forget;
    EXPECT_FLOAT_EQ(0.23, activation_alpha[0]);
    EXPECT_FLOAT_EQ(0.33, activation_beta[0]);
    EXPECT_EQ(ppl::nn::onnx::LSTMParam::ACT_ELU, activations[0]);
    EXPECT_FLOAT_EQ(0.34, clip);
    EXPECT_EQ(ppl::nn::onnx::LSTMParam::DIR_REVERSE, direction);
    EXPECT_EQ(44, hidden_size);
    EXPECT_EQ(23, input_forget);
}
