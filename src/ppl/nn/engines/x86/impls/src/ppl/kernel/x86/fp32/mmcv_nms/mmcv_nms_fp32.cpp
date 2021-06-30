#include "ppl/kernel/x86/common/internal_include.h"

#include <vector>
#include <algorithm>
#include <numeric>

namespace ppl { namespace kernel { namespace x86 {

inline float calc_iou(
        const float *boxes,
        const float *areas,
        const int64_t i0,
        const int64_t i1,
        const int64_t offset)
{
    float ix1 = boxes[i0 * 4 + 0];
    float iy1 = boxes[i0 * 4 + 1];
    float ix2 = boxes[i0 * 4 + 2];
    float iy2 = boxes[i0 * 4 + 3];

    float xx1 = max(ix1, boxes[i1 * 4 + 0]);
    float yy1 = max(iy1, boxes[i1 * 4 + 1]);
    float xx2 = min(ix2, boxes[i1 * 4 + 2]);
    float yy2 = min(iy2, boxes[i1 * 4 + 3]);

    float w = max(0.f, xx2 - xx1 + offset);
    float h = max(0.f, yy2 - yy1 + offset);

    float inter = w * h;
    float ovr = inter / (areas[i0] + areas[i1] - inter);
    return ovr;
}

ppl::common::RetCode mmcv_nms_ndarray_fp32_naive(
        const float *boxes,
        const float *scores,
        const uint32_t num_boxes_in,
        const float iou_threshold,
        const int64_t offset,
        int64_t *dst,
        int64_t *num_boxes_out)
{
    std::vector<uint32_t> sorted_index_(num_boxes_in);
    std::vector<float> areas_(num_boxes_in);
    uint32_t *sorted_index = sorted_index_.data();
    float *areas = areas_.data();

    for (uint32_t i = 0; i < num_boxes_in; i++) {
        areas[i] = (boxes[i * 4 + 2] - boxes[i * 4 + 0] + offset) * (boxes[i * 4 + 3] - boxes[i * 4 + 1] + offset);
    }
    argsort(scores, sorted_index, num_boxes_in);

    *num_boxes_out = 0;
    for (uint32_t i = 0; i < num_boxes_in; i++) {
        int64_t idx = sorted_index[i];
        bool keep = true;
        for (int64_t j = 0; j < *num_boxes_out; j++) {
            float iou = calc_iou(boxes, areas, idx, dst[j], offset);
            if (iou >= iou_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            dst[(*num_boxes_out)++] = idx;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode mmcv_nms_ndarray_fp32(
        const float *boxes,
        const float *scores,
        const uint32_t num_boxes_in,
        const float iou_threshold,
        const int64_t offset,
        int64_t *dst,
        int64_t *num_boxes_out)
{
    return mmcv_nms_ndarray_fp32_naive(boxes, scores, num_boxes_in, iou_threshold, offset, dst, num_boxes_out);
}

}}}; // namespace ppl::kernel::x86
