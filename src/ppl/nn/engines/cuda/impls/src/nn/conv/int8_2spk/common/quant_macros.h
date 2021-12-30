#define cvtOutData(_outData,cvtData0,cvtData1,cvtData2,cvtData3,cvtData4,cvtData5,cvtData6,cvtData7){ \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(cvtData4) : "r"(cvtData6), "r"(cvtData4)); \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(cvtData5) : "r"(cvtData7), "r"(cvtData5)); \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(cvtData0) : "r"(cvtData2), "r"(cvtData0), "r"(cvtData4)); \
	    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(cvtData1) : "r"(cvtData3), "r"(cvtData1), "r"(cvtData5)); \
	    _outData.x = cvtData0; _outData.y = cvtData1; \
}

/*,intMin,intMax){ */
#define quantOutData(_C, _fC, _outInt8Scale){ \
	   _C.x = __float2int_rn(_fC[0]*_outInt8Scale); \
	   _C.y = __float2int_rn(_fC[1]*_outInt8Scale); \
	   _C.z = __float2int_rn(_fC[2]*_outInt8Scale); \
	   _C.w = __float2int_rn(_fC[3]*_outInt8Scale); \
}


#define deQuantData(_fC, _Rv4, _descale){ \
	    _fC[0] = _Rv4.x * _descale.x; \
	    _fC[1] = _Rv4.y * _descale.y; \
	    _fC[2] = _Rv4.z * _descale.z; \
	    _fC[3] = _Rv4.w * _descale.w; \
}

#define LOAD_SCALE(_scale, _d_scale){ \
    if( dCv4_x_valid && dCv4_y_valid ){ \
        _scale = ((float4*)_d_scale)[grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
        _scale.x *= in_scale; \
        _scale.y *= in_scale; \
        _scale.z *= in_scale; \
        _scale.w *= in_scale; \
    } \
}

#define packchar4(_outData, x, y, z, w){ \
    if (x>127)	x = 127;                 \
    if (x<-128) x = -128;                \
    if (y>127)	y = 127;                 \
    if (y<-128) y = -128;                \
    if (z>127)	z = 127;                 \
    if (z<-128) z = -128;                \
    if (w>127)	w = 127;                 \
    if (w<-128) w = -128;                \
    x = (0xffu & (int8_t)x);             \
    y = (0xffu & (int8_t)y) << 8;        \
    z = (0xffu & (int8_t)z) << 16;       \
    w = (0xffu & (int8_t)w) << 24;         \
    _outData = w | z | y | x;/*(x,y,z,w)*/\
}

