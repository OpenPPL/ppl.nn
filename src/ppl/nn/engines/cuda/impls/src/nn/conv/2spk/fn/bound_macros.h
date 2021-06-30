////////////////////////////////////////
// filter shifting macros
////////////////////////////////////////

#define FWD_FLT_SIZE1(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \
        { \
            _flt_w_id++; \
            in_w_id[0] += hole_width; \
            \
            if(_flt_w_id == flt_width) \
            {\
                _flt_w_id = 0; \
                in_w_id[0] = in_w_start[0]; \
                _flt_h_id++; \
                in_h_id[0] += hole_height; \
            } \
            \
            if(_flt_h_id == flt_height) \
            { \
                _flt_h_id = 0;   \
                in_h_id[0] = in_h_start[0]; \
                \
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \
                \
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \
            } \
        }

#define FWD_FLT_SIZE2(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \
        { \
            _flt_w_id++; \
            in_w_id[0] += hole_width;        in_w_id[1] += hole_width; \
            \
            if(_flt_w_id == flt_width) \
            {\
                _flt_w_id = 0; \
                in_w_id[0] = in_w_start[0];  in_w_id[1] = in_w_start[1]; \
                _flt_h_id++; \
                in_h_id[0] += hole_height;   in_h_id[1] += hole_height; \
            } \
            \
            if(_flt_h_id == flt_height) \
            { \
                _flt_h_id = 0;   \
                in_h_id[0] = in_h_start[0];  in_h_id[1] = in_h_start[1]; \
                \
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \
                \
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \
            } \
        }

#define FWD_FLT_SIZE4(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \
        { \
            _flt_w_id++; \
            in_w_id[0] += hole_width;        in_w_id[1] += hole_width;   in_w_id[2] += hole_width;  in_w_id[3] += hole_width; \
            \
            if(_flt_w_id == flt_width) \
            { \
                _flt_w_id = 0; \
                in_w_id[0] = in_w_start[0];  in_w_id[1] = in_w_start[1]; in_w_id[2] = in_w_start[2];  in_w_id[3] = in_w_start[3]; \
                _flt_h_id++; \
                in_h_id[0] += hole_height;   in_h_id[1] += hole_height;  in_h_id[2] += hole_height;   in_h_id[3] += hole_height; \
            } \
            \
            if(_flt_h_id == flt_height) \
            { \
                _flt_h_id = 0;   \
                in_h_id[0] = in_h_start[0];  in_h_id[1] = in_h_start[1]; in_h_id[2] = in_h_start[2];  in_h_id[3] = in_h_start[3]; \
                \
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \
                \
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \
            } \
        }

#define FWD_FLT_SIZE8(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \
        { \
            _flt_w_id++; \
            in_w_id[0] += hole_width;        in_w_id[1] += hole_width;   in_w_id[2] += hole_width;  in_w_id[3] += hole_width; \
            in_w_id[4] += hole_width;        in_w_id[5] += hole_width;   in_w_id[6] += hole_width;  in_w_id[7] += hole_width; \
            \
            if(_flt_w_id == flt_width) \
            { \
                _flt_w_id = 0; \
                in_w_id[0] = in_w_start[0];  in_w_id[1] = in_w_start[1]; in_w_id[2] = in_w_start[2];  in_w_id[3] = in_w_start[3]; \
                in_w_id[4] = in_w_start[4];  in_w_id[5] = in_w_start[5]; in_w_id[6] = in_w_start[6];  in_w_id[7] = in_w_start[7]; \
                _flt_h_id++; \
                in_h_id[0] += hole_height;   in_h_id[1] += hole_height;  in_h_id[2] += hole_height;   in_h_id[3] += hole_height; \
                in_h_id[4] += hole_height;   in_h_id[5] += hole_height;  in_h_id[6] += hole_height;   in_h_id[7] += hole_height; \
            } \
            \
            if(_flt_h_id == flt_height) \
            { \
                _flt_h_id = 0;   \
                in_h_id[0] = in_h_start[0];  in_h_id[1] = in_h_start[1]; in_h_id[2] = in_h_start[2];  in_h_id[3] = in_h_start[3]; \
                in_h_id[4] = in_h_start[4];  in_h_id[5] = in_h_start[5]; in_h_id[6] = in_h_start[6];  in_h_id[7] = in_h_start[7]; \
                \
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \
                \
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \
            } \
        }

#define FWD_FLT_SIZE16(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \
        { \
            _flt_w_id++; \
            in_w_id[0]  += hole_width;        in_w_id[1]  += hole_width;   in_w_id[2]  += hole_width;  in_w_id[3]  += hole_width; \
            in_w_id[4]  += hole_width;        in_w_id[5]  += hole_width;   in_w_id[6]  += hole_width;  in_w_id[7]  += hole_width; \
            in_w_id[8]  += hole_width;        in_w_id[9]  += hole_width;   in_w_id[10] += hole_width;  in_w_id[11] += hole_width; \
            in_w_id[12] += hole_width;        in_w_id[13] += hole_width;   in_w_id[14] += hole_width;  in_w_id[15] += hole_width; \
            \
            if(_flt_w_id == flt_width) \
            { \
                _flt_w_id = 0; \
                in_w_id[0]  = in_w_start[0];   in_w_id[1]  = in_w_start[1];  in_w_id[2]  = in_w_start[2];   in_w_id[3]  = in_w_start[3]; \
                in_w_id[4]  = in_w_start[4];   in_w_id[5]  = in_w_start[5];  in_w_id[6]  = in_w_start[6];   in_w_id[7]  = in_w_start[7]; \
                in_w_id[8]  = in_w_start[8];   in_w_id[9]  = in_w_start[9];  in_w_id[10] = in_w_start[10];  in_w_id[11] = in_w_start[11]; \
                in_w_id[12] = in_w_start[12];  in_w_id[13] = in_w_start[13]; in_w_id[14] = in_w_start[14];  in_w_id[15] = in_w_start[15]; \
                _flt_h_id++; \
                in_h_id[0]  += hole_height;        in_h_id[1]  += hole_height;   in_h_id[2]  += hole_height;  in_h_id[3]  += hole_height; \
                in_h_id[4]  += hole_height;        in_h_id[5]  += hole_height;   in_h_id[6]  += hole_height;  in_h_id[7]  += hole_height; \
                in_h_id[8]  += hole_height;        in_h_id[9]  += hole_height;   in_h_id[10] += hole_height;  in_h_id[11] += hole_height; \
                in_h_id[12] += hole_height;        in_h_id[13] += hole_height;   in_h_id[14] += hole_height;  in_h_id[15] += hole_height; \
            } \
            \
            if(_flt_h_id == flt_height) \
            { \
                _flt_h_id = 0;   \
                in_h_id[0]  = in_h_start[0];   in_h_id[1]  = in_h_start[1];  in_h_id[2]  = in_h_start[2];   in_h_id[3]  = in_h_start[3]; \
                in_h_id[4]  = in_h_start[4];   in_h_id[5]  = in_h_start[5];  in_h_id[6]  = in_h_start[6];   in_h_id[7]  = in_h_start[7]; \
                in_h_id[8]  = in_h_start[8];   in_h_id[9]  = in_h_start[9];  in_h_id[10] = in_h_start[10];  in_h_id[11] = in_h_start[11]; \
                in_h_id[12] = in_h_start[12];  in_h_id[13] = in_h_start[13]; in_h_id[14] = in_h_start[14];  in_h_id[15] = in_h_start[15]; \
                \
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \
                \
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \
            } \
        }
