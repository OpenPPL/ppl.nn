#define cvtOutData(_outData,cvtData0,cvtData1,cvtData2,cvtData3,cvtData4,cvtData5,cvtData6,cvtData7){ \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(cvtData4) : "r"(cvtData6), "r"(cvtData4)); \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(cvtData5) : "r"(cvtData7), "r"(cvtData5)); \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(cvtData0) : "r"(cvtData2), "r"(cvtData0), "r"(cvtData4)); \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(cvtData1) : "r"(cvtData3), "r"(cvtData1), "r"(cvtData5)); \
	    _outData.x = cvtData0; _outData.y = cvtData1; \
}

/*,intMin,intMax){ */
#define quantOutData_x1(_C, _fCv2, _outInt8Scale){ \
	   if(dCv1_y_valid[0] && dCv1_x_valid[0]) _C[Cv1_off + 0].x = __float2int_rn(_fCv2[Cv1_off + 0].x*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[0]) _C[Cv1_off + 0].y = __float2int_rn(_fCv2[Cv1_off + 0].y*_outInt8Scale); \
}
#define quantOutData_x2(_C, _fCv2, _outInt8Scale){ \
	   if(dCv1_y_valid[0] && dCv1_x_valid[0]) _C[Cv1_off + 0].x = __float2int_rn(_fCv2[Cv1_off + 0].x*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[0]) _C[Cv1_off + 0].y = __float2int_rn(_fCv2[Cv1_off + 0].y*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[1]) _C[Cv1_off + 1].x = __float2int_rn(_fCv2[Cv1_off + 1].x*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[1]) _C[Cv1_off + 1].y = __float2int_rn(_fCv2[Cv1_off + 1].y*_outInt8Scale); \
}
#define quantOutData_x4(_C, _fCv2, _outInt8Scale){ \
	   if(dCv1_y_valid[0] && dCv1_x_valid[0]) _C[Cv1_off + 0].x = __float2int_rn(_fCv2[Cv1_off + 0].x*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[0]) _C[Cv1_off + 0].y = __float2int_rn(_fCv2[Cv1_off + 0].y*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[1]) _C[Cv1_off + 1].x = __float2int_rn(_fCv2[Cv1_off + 1].x*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[1]) _C[Cv1_off + 1].y = __float2int_rn(_fCv2[Cv1_off + 1].y*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[2]) _C[Cv1_off + 2].x = __float2int_rn(_fCv2[Cv1_off + 2].x*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[2]) _C[Cv1_off + 2].y = __float2int_rn(_fCv2[Cv1_off + 2].y*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[3]) _C[Cv1_off + 3].x = __float2int_rn(_fCv2[Cv1_off + 3].x*_outInt8Scale); \
	   if(dCv1_y_valid[0] && dCv1_x_valid[3]) _C[Cv1_off + 3].y = __float2int_rn(_fCv2[Cv1_off + 3].y*_outInt8Scale); \
}

#define deQuantData_x1(_fCv2, _Rv2, _descale){ \
	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) _fCv2[Cv1_off + 0].x = _Rv2[Cv1_off + 0].x * _descale[0].x; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) _fCv2[Cv1_off + 0].y = _Rv2[Cv1_off + 0].y * _descale[0].y; \
}
#define deQuantData_x2(_fCv2, _Rv2, _descale){ \
	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) _fCv2[Cv1_off + 0].x = _Rv2[Cv1_off + 0].x * _descale[0].x; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) _fCv2[Cv1_off + 0].y = _Rv2[Cv1_off + 0].y * _descale[0].y; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) _fCv2[Cv1_off + 1].x = _Rv2[Cv1_off + 1].x * _descale[1].x; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) _fCv2[Cv1_off + 1].y = _Rv2[Cv1_off + 1].y * _descale[1].y; \
}
#define deQuantData_x4(_fCv2, _Rv2, _descale){ \
	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) _fCv2[Cv1_off + 0].x = _Rv2[Cv1_off + 0].x * _descale[0].x; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) _fCv2[Cv1_off + 0].y = _Rv2[Cv1_off + 0].y * _descale[0].y; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) _fCv2[Cv1_off + 1].x = _Rv2[Cv1_off + 1].x * _descale[1].x; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) _fCv2[Cv1_off + 1].y = _Rv2[Cv1_off + 1].y * _descale[1].y; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[2]) _fCv2[Cv1_off + 2].x = _Rv2[Cv1_off + 2].x * _descale[2].x; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[2]) _fCv2[Cv1_off + 2].y = _Rv2[Cv1_off + 2].y * _descale[2].y; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[3]) _fCv2[Cv1_off + 3].x = _Rv2[Cv1_off + 3].x * _descale[3].x; \
	    if(dCv1_y_valid[0] && dCv1_x_valid[3]) _fCv2[Cv1_off + 3].y = _Rv2[Cv1_off + 3].y * _descale[3].y; \
}

#define LOAD_SCALE_x1(_scale, _d_scale){ \
    if( dCv1_x_valid[0] && dCv1_y_valid[0] ){ \
        _scale[0] = ((float2*)_d_scale)[dCv1_idx[0]]; \
        _scale[0].x *= in_scale; \
        _scale[0].y *= in_scale; \
    } \
}
#define LOAD_SCALE_x2(_scale, _d_scale){ \
    if( dCv1_x_valid[0] && dCv1_y_valid[0] ) \
        _scale[0] = ((float2*)_d_scale)[dCv1_idx[0]]; \
    if( dCv1_x_valid[1] && dCv1_y_valid[0] ) \
        _scale[1] = ((float2*)_d_scale)[dCv1_idx[1]]; \
/*
if (dCv1_idy[0] * num_flt_v2 + dCv1_idx[0] == 32/4)    printf("init kernelout inscale: %d, %f %f\t", dCv1_idx[0]*2, _scale[0].x, _scale[0].y); \
if(tid==0 && blockIdx.z==0 && blockIdx.x==0 && blockIdx.z==0) {\
    printf("flt scale: %x\n", _d_scale); \
    for(int i=0; i<96*16; i++) \
    if(((float*)d_flt_scale)[i]!=0.f)    printf("%d:%f, ", i, ((float*)d_flt_scale)[i]); \
}\
*/\
    if( dCv1_x_valid[0] && dCv1_y_valid[0] ){ \
        _scale[0].x *= in_scale; \
        _scale[0].y *= in_scale; \
    } \
    if( dCv1_x_valid[1] && dCv1_y_valid[0] ){ \
        _scale[1].x *= in_scale; \
        _scale[1].y *= in_scale; \
    } \
/*
if (dCv1_idy[0] * num_flt_v2 + dCv1_idx[0] == 32/4)    printf("kernelout inscale: %f \t", _scale[0].x); \
*/\
}
#define LOAD_SCALE_x4(_scale, _d_scale){ \
    if( dCv1_x_valid[0] && dCv1_y_valid[0] ) \
        _scale[0] = ((float2*)_d_scale)[dCv1_idx[0]]; \
    if( dCv1_x_valid[1] && dCv1_y_valid[0] ) \
        _scale[1] = ((float2*)_d_scale)[dCv1_idx[1]]; \
    if( dCv1_x_valid[2] && dCv1_y_valid[0] ) \
        _scale[2] = ((float2*)_d_scale)[dCv1_idx[2]]; \
    if( dCv1_x_valid[3] && dCv1_y_valid[0] ) \
        _scale[3] = ((float2*)_d_scale)[dCv1_idx[3]]; \
    if( dCv1_x_valid[0] && dCv1_y_valid[0] ){ \
        _scale[0].x *= in_scale; \
        _scale[0].y *= in_scale; \
    } \
    if( dCv1_x_valid[1] && dCv1_y_valid[0] ){ \
        _scale[1].x *= in_scale; \
        _scale[1].y *= in_scale; \
    } \
    if( dCv1_x_valid[2] && dCv1_y_valid[0] ){ \
        _scale[2].x *= in_scale; \
        _scale[2].y *= in_scale; \
    } \
    if( dCv1_x_valid[3] && dCv1_y_valid[0] ){ \
        _scale[3].x *= in_scale; \
        _scale[3].y *= in_scale; \
    } \
}

#define packChar2(_outData, _Cv2){ \
    if (_Cv2.x>127)	_Cv2.x = 127;                 \
    if (_Cv2.x<-128)    _Cv2.x = -128;                \
    if (_Cv2.y>127)	_Cv2.y = 127;                 \
    if (_Cv2.y<-128)    _Cv2.y = -128;                \
    _Cv2.x = (0xffu & (int8_t)_Cv2.x);             \
    _Cv2.y = (0xffu & (int8_t)_Cv2.y) << 8;        \
    _outData = _Cv2.y | _Cv2.x;/*(x,y,z,w)*/\
}

#define packChar2_x1(_outData, _Cv2){ \
	packChar2(_outData[0], _Cv2[Cv1_off + 0]); \
}
#define packChar2_x2(_outData, _Cv2){ \
	packChar2(_outData[0], _Cv2[Cv1_off + 0]); \
	packChar2(_outData[1], _Cv2[Cv1_off + 1]); \
}
#define packChar2_x4(_outData, _Cv2){ \
	packChar2(_outData[0], _Cv2[Cv1_off + 0]); \
	packChar2(_outData[1], _Cv2[Cv1_off + 1]); \
	packChar2(_outData[2], _Cv2[Cv1_off + 2]); \
	packChar2(_outData[3], _Cv2[Cv1_off + 3]); \
}
