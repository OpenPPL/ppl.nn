#define SET_BOUND_FLT3(_in_hw_mask, _in_n_id, _in_h_id, _in_w_id) \
        { \
            if(_in_n_id < in_num) \
            { \
                _in_hw_mask = 0xffffffff; \
                if(_in_h_id < 0 || _in_h_id >= in_height) _in_hw_mask = _in_hw_mask & 0xfffffff8; \
                if(_in_w_id < 0 || _in_w_id >= in_width)  _in_hw_mask = _in_hw_mask & 0xffffffb6; \
                \
                _in_h_id += hole_height; \
                _in_w_id += hole_width; \
                \
                if(_in_h_id < 0 || _in_h_id >= in_height) _in_hw_mask = _in_hw_mask & 0xffffffc7; \
                if(_in_w_id < 0 || _in_w_id >= in_width)  _in_hw_mask = _in_hw_mask & 0xffffff6d; \
                \
                _in_h_id += hole_height; \
                _in_w_id += hole_width; \
                \
                if(_in_h_id < 0 || _in_h_id >= in_height)  _in_hw_mask = _in_hw_mask & 0xfffffe3f; \
                if(_in_w_id < 0 || _in_w_id >= in_width)   _in_hw_mask = _in_hw_mask & 0xfffffedb; \
            } else { \
                _in_hw_mask = 0x0; \
            } \
        }

#define FWD_FLT3(_flt_hw_id, _flt_hw_bid, _flt_c_v8_id, _flt_c_v8_valid) \
        { \
            if(_flt_hw_id == 8) \
            { \
                _flt_hw_id = 0; \
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \
                \
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \
            } else { \
                _flt_hw_id = _flt_hw_id + 1; \
            } \
            \
            _flt_hw_bid = (0x1 << _flt_hw_id); \
        }

#define FWD_FLT(_flt_hw_id, _flt_hw_bid, _flt_c_v8_id, _flt_c_v8_valid)    FWD_FLT3(_flt_hw_id, _flt_hw_bid, _flt_c_v8_id, _flt_c_v8_valid)
