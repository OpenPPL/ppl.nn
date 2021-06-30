#define SET_BOUND_FLT1(_in_hw_mask, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _in_hw_mask = _in_n_id <  in_num && \
                        _in_h_id >= 0 && _in_h_id < in_height && \
                        _in_w_id >= 0 && _in_w_id < in_width; \
        }

#define FWD_FLT1(_flt_c_v8_id, _flt_c_v8_valid) \
        { \
            flt_c_v8_id   += TILE_K_V8_PER_CTA; \
            _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \
        }

#define FWD_FLT(_flt_c_v8_id, _flt_c_v8_valid)    FWD_FLT1(_flt_c_v8_id, _flt_c_v8_valid)
