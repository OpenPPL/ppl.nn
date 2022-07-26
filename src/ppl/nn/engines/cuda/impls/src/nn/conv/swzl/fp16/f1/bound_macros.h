// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

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
