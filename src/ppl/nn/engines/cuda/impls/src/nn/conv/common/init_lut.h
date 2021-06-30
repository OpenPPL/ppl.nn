#ifndef __PPLCUDA_INIT_LUT_H__
#define __PPLCUDA_INIT_LUT_H__

#define MAX_LUT_SIZE  128

#define MAX_SPLITK_SIZE 8

struct lut_t{
    int idx[MAX_LUT_SIZE];
    lut_t(){};
};

struct chl_lut_t{
    int idx[MAX_SPLITK_SIZE + 1];
};

struct kloop_lut_t{
    int idx[MAX_SPLITK_SIZE + 1];
};

void InitializeInputLut(
        int& in_lut_size,
        int* in_lut,
        int  flt_height,
        int  flt_width,
        int  in_height,
        int  in_width,
        int  pad_height,
        int  pad_width,
        int  hole_height,
        int  hole_width,
        int  num_chl_per_grp_pad,
        int  num_grp,
        int  num_chl_per_step,
        int  pad_size);


void InitializeFilterLut(
        int& flt_lut_size,
        int* flt_lut,
        int  flt_height,
        int  flt_width,
        int  num_chl_per_grp_pad,
        int  num_chl_per_step,
        int  pad_size);

void InitializeAbsChlLut(
        int& abs_chl_lut_size,
        int* abs_chl_lut,
        int  num_chl,
        int  num_grp,
        int  pad_size,
        int  num_chl_per_step,
        int  splitk);

void InitializeChlLut(
        int& chl_lut_size,
        int* chl_lut,
        int  num_chl,
        int  num_grp,
        int  pad_size,
        int  num_chl_per_step,
        int  splitk);

void InitializeKloopLut(
        int& kloop_lut_size,
        int* kloop_lut,
        int  num_chl,
        int  num_grp,
        int  pad_size,
        int  num_chl_per_step,
        int  splitk,
        int  splitf,
        int  flt_hw);

#endif
