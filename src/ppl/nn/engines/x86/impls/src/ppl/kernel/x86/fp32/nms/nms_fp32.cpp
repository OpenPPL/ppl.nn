
#include <algorithm>
#include <vector>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

inline float calc_iou(const float *p_boxes, int64_t i0, int64_t i1, bool centered)
{
    const float *b0 = p_boxes + i0 * 4;
    const float *b1 = p_boxes + i1 * 4;
    float x_min, x_max, y_min, y_max;
    float w0, w1, h0, h1;
    if (centered == true) { // tf_format: [x_center, y_center, width, height]
        w0    = b0[2];
        w1    = b1[2];
        h0    = b0[3];
        h1    = b1[3];
        x_min = min(b0[0] - w0 / 2, b1[0] - w1 / 2);
        x_max = max(b0[0] + w0 / 2, b1[0] + w1 / 2);
        y_min = min(b0[1] - h0 / 2, b1[1] - h1 / 2);
        y_max = max(b0[1] + h0 / 2, b1[1] + h1 / 2);
    } else { // pytorch_format: [y1, x1, y2, x2]
        w0    = abs(b0[1] - b0[3]);
        w1    = abs(b1[1] - b1[3]);
        h0    = abs(b0[0] - b0[2]);
        h1    = abs(b1[0] - b1[2]);
        x_min = min(min(b0[1], b0[3]), min(b1[1], b1[3]));
        x_max = max(max(b0[1], b0[3]), max(b1[1], b1[3]));
        y_min = min(min(b0[0], b0[2]), min(b1[0], b1[2]));
        y_max = max(max(b0[0], b0[2]), max(b1[0], b1[2]));
    }

    if (w0 + w1 <= x_max - x_min || h0 + h1 <= y_max - y_min) {
        return 0;
    }

    float area0 = w0 * h0, area1 = w1 * h1;
    float iw = w0 + w1 - (x_max - x_min);
    float ih = h0 + h1 - (y_max - y_min);
    float I  = ih * iw;
    float U  = area0 + area1 - I;

    return I / U;
}

static ppl::common::RetCode nms_ndarray_naive(
    const float *boxes,
    const float *scommons,
    const uint32_t num_boxes_in,
    const uint32_t batch,
    const uint32_t num_classes,
    const bool center_point_box,
    const int64_t maxoutput_boxes_per_batch_per_class,
    const float iou_threshold,
    const float scommon_threshold,
    int64_t *dst,
    int64_t *num_boxes_out)
{
    std::vector<uint32_t> sorted_index_;
    std::vector<uint32_t> selected_index_;
    sorted_index_.resize(num_boxes_in);
    selected_index_.resize(num_boxes_in);
    uint32_t *sorted_index   = sorted_index_.data();
    uint32_t *selected_index = selected_index_.data();

    uint64_t out_idx = 0;
    for (int64_t n = 0; n < batch; n++) {
        const float *p_boxes = boxes + n * num_boxes_in * 4;
        for (int64_t c = 0; c < num_classes; c++) {
            int64_t selected_num   = 0;
            const float *p_scommons = scommons + (n * num_classes + c) * num_boxes_in;
            argsort(p_scommons, sorted_index, num_boxes_in);
            for (int64_t i = 0; i < num_boxes_in; i++) {
                int64_t idx = sorted_index[i];
                if (p_scommons[idx] <= scommon_threshold) {
                    break;
                }
                bool keep = true;
                for (int64_t j = 0; j < selected_num; j++) {
                    float iou = calc_iou(p_boxes, idx, selected_index[j], center_point_box);
                    if (iou > iou_threshold) {
                        keep = false;
                        break;
                    }
                }
                if (keep) { // box i has not been suppressed
                    selected_index[selected_num++] = idx;
                }
                if (selected_num >= maxoutput_boxes_per_batch_per_class) {
                    break;
                }
            }
            // process result
            for (int64_t i = 0; i < selected_num; i++) {
                int64_t *p_dst = dst + out_idx * 3;

                p_dst[0] = n;
                p_dst[1] = c;
                p_dst[2] = selected_index[i];
                out_idx++;
            }
        }
    }

    *num_boxes_out = out_idx;
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode nms_ndarray_fp32(
    const float *boxes,
    const float *scommons,
    const uint32_t num_boxes_in,
    const uint32_t batch,
    const uint32_t num_classes,
    const bool center_point_box,
    const int64_t maxoutput_boxes_per_batch_per_class,
    const float iou_threshold,
    const float scommon_threshold,
    int64_t *dst,
    int64_t *num_boxes_out)
{
    return nms_ndarray_naive(boxes, scommons, num_boxes_in, batch, num_classes, center_point_box, maxoutput_boxes_per_batch_per_class, iou_threshold, scommon_threshold, dst, num_boxes_out);
}

}}}; // namespace ppl::kernel::x86
