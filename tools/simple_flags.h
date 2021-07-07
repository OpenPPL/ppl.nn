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

#ifndef SIMPLE_FLAGS_H
#define SIMPLE_FLAGS_H
#include <stdint.h>
#include <vector>
#include <string>

#define BEGIN_FLAGS_NAMESPACES namespace simple_flags {
#define END_FLAGS_NAMESPACES }

BEGIN_FLAGS_NAMESPACES
typedef std::string flag_string;
typedef bool flag_bool;
typedef float flag_float;
typedef double flag_double;
typedef int16_t flag_int16;
typedef uint16_t flag_uint16;
typedef int32_t flag_int32;
typedef uint32_t flag_uint32;
typedef int64_t flag_int64;
typedef uint64_t flag_uint64;
typedef std::vector<flag_string> flag_stringlist;
typedef std::vector<flag_bool> flag_boollist;
typedef std::vector<flag_float> flag_floatlist;
typedef std::vector<flag_double> flag_doublelist;
typedef std::vector<flag_int16> flag_int16list;
typedef std::vector<flag_int32> flag_int32list;
typedef std::vector<flag_int64> flag_int64list;
typedef std::vector<flag_uint16> flag_uint16list;
typedef std::vector<flag_uint32> flag_uint32list;
typedef std::vector<flag_uint64> flag_uint64list;

int parse_args(int argc, char** argv);
void print_args_info();

template <typename T>
void registerFlag(const flag_string& opt, T* optPtr, const char* comment);

const flag_stringlist& get_unknown_flags();

END_FLAGS_NAMESPACES

/**
 * \brief Flag_To_Str
 * Convert the given literal expression to literal string.
 * \param x literal expression to be converted.
 */
#define Flag_To_Str(x) #x

/**
 * \defgroup Single-parameter flag declaration macros
 * Declare a option which accept only one parameter, so you can access it by using Flag_* vairable.
 * @{
 */
#define Declare_bool(opt) extern simple_flags::flag_bool Flag_##opt
#define Declare_float(opt) extern simple_flags::flag_float Flag_##opt
#define Declare_double(opt) extern simple_flags::flag_double Flag_##opt
#define Declare_int32(opt) extern simple_flags::flag_int32 Flag_##opt
#define Declare_uint32(opt) extern simple_flags::flag_uint32 Flag_##opt
#define Declare_int64(opt) extern simple_flags::flag_int64 Flag_##opt
#define Declare_uint64(opt) extern simple_flags::flag_uint64 Flag_##opt
#define Declare_string(opt) extern simple_flags::flag_string Flag_##opt

/**
 * @} //! Common declare macros.
 */

/**
 * \defgroup Muti-parameter flag declaration macros
 * Declare a option which accept parameters list, so you can access it by using Flag_* variable
 */
#define Declare_stringlist(opt) extern simple_flags::flag_stringlist Flag_##opt
#define Declare_boollist(opt) extern simple_flags::flag_boollist Flag_##opt
#define Declare_floatlist(opt) extern simple_flags::flag_floatlist Flag_##opt
#define Declare_doublelist(opt) extern simple_flags::flag_doublelist Flag_##opt
#define Declare_int32list(opt) extern simple_flags::flag_int32list Flag_##opt
#define Declare_int64list(opt) extern simple_flags::flag_int64list Flag_##opt
#define Declare_uint32list(opt) extern simple_flags::flag_uint32list Flag_##opt
#define Declare_uint64list(opt) extern simple_flags::flag_uint64list Flag_##opt

/**@}*/

/**
 * \defgroup Self-defined single-parameter flag declaration macros
 * Declare a self-defined option which accept only one parameter, so you can access it by using Flag_* variable
 * @{
 */
#define Declare_bool_opt(flag) extern simple_flags::flag_bool flag
#define Declare_float_opt(flag) extern simple_flags::flag_float flag
#define Declare_double_opt(flag) extern simple_flags::flag_double flag
#define Declare_int32_opt(flag) extern simple_flags::flag_int32 flag
#define Declare_uint32_opt(flag) extern simple_flags::flag_uint32 flag
#define Declare_int64_opt(flag) extern simple_flags::flag_int64 flag
#define Declare_uint64_opt(flag) extern simple_flags::flag_uint64 flag
#define Declare_string_opt(flag) extern simple_flags::flag_string flag

/**@}*/

/**
 * \defgroup Self-defined multi-parameter flag declaration macros
 * Declare a self-define option which accept parameter list, so you can access it by using Flag_* variable.
 * @{
 */
#define Declare_stringlist_opt(flag) extern simple_flags::flag_stringlist flag
#define Declare_boollist_opt(flag) extern simple_flags::flag_boollist flag
#define Declare_floatlist_opt(flag) extern simple_flags::flag_floatlist flag
#define Declare_doublelist_opt(flag) extern simple_flags::flag_doublelist flag
#define Declare_int32list_opt(flag) extern simple_flags::flag_int32list flag
#define Declare_int64list_opt(flag) extern simple_flags::flag_int64list flag
#define Declare_uint32list_opt(flag) extern simple_flags::flag_uint32list flag
#define Declare_uint64list_opt(flag) extern simple_flags::flag_uint64list flag

/**@}*/

/**
 * \defgroup Define definitions
 */
#define Define_Implementer(type, opt, def, comment)                         \
    simple_flags::type Flag_##opt = def;                                    \
    BEGIN_FLAGS_NAMESPACES                                                  \
    class type##_Flag_Register_##opt {                                      \
    public:                                                                 \
        type##_Flag_Register_##opt() {                                      \
            registerFlag<type>("-" Flag_To_Str(opt), &Flag_##opt, comment); \
        }                                                                   \
    };                                                                      \
    static type##_Flag_Register_##opt s_flag_##opt##_object;                \
    END_FLAGS_NAMESPACES

#define Define_ImplementerOpt(type, opt, flag, def, comment)                 \
    simple_flags::type flag = def;                                           \
    BEGIN_FLAGS_NAMESPACES                                                   \
    class type##_Flag_Register_##flag {                                      \
    public:                                                                  \
        type##_Flag_Register_##flag() {                                      \
            registerFlag<type>(opt, &flag, comment);                         \
        }                                                                    \
    };                                                                       \
    static type##_Flag_Register_##flag type##_Flag_Register_##flag##_object; \
    END_FLAGS_NAMESPACES

#define Define_Implementer_list(type, opt, comment)                         \
    simple_flags::type Flag_##opt;                                          \
    BEGIN_FLAGS_NAMESPACES                                                  \
    class type##_Flag_Register_##opt {                                      \
    public:                                                                 \
        type##_Flag_Register_##opt() {                                      \
            registerFlag<type>("-" Flag_To_Str(opt), &Flag_##opt, comment); \
        }                                                                   \
    };                                                                      \
    static type##_Flag_Register_##opt type##_Flag_Register_##opt##_object;  \
    END_FLAGS_NAMESPACES

#define Define_Implementer_listOpt(type, opt, flag, comment)                 \
    simple_flags::type flag;                                                 \
    BEGIN_FLAGS_NAMESPACES                                                   \
    class type##_Flag_Register_##flag {                                      \
    public:                                                                  \
        type##_Flag_Register_##flag() {                                      \
            registerFlag<type>(opt, &flag, comment);                         \
        }                                                                    \
    };                                                                       \
    static type##_Flag_Register_##flag type##_Flag_Register_##flag##_object; \
    END_FLAGS_NAMESPACES

/**@}*/

/**
 * \defgroup Single-parameter option define macros
 * Define a option which accept only one option, so it will be parsed.
 * @{
 */
#define Define_bool(opt, def, comment) Define_Implementer(flag_bool, opt, def, comment)
#define Define_float(opt, def, comment) Define_Implementer(flag_float, opt, def, comment)
#define Define_double(opt, def, comment) Define_Implementer(flag_double, opt, def, comment)
#define Define_int32(opt, def, comment) Define_Implementer(flag_int32, opt, def, comment)
#define Define_uint32(opt, def, comment) Define_Implementer(flag_uint32, opt, def, comment)
#define Define_int64(opt, def, comment) Define_Implementer(flag_int64, opt, def, comment)
#define Define_uint64(opt, def, comment) Define_Implementer(flag_uint64, opt, def, comment)
#define Define_string(opt, def, comment) Define_Implementer(flag_string, opt, def, comment)

/**@}*/

/**
 * \defgroup Self-define single-parameter option define macros
 * Define self-defined optiong, so it will be parsed.
 * @{
 */
#define Define_bool_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_bool, opt, flag, def, comment)
#define Define_float_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_float, opt, flag, def, comment)
#define Define_double_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_double, opt, flag, def, comment)
#define Define_int32_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_int32, opt, flag, def, comment)
#define Define_uint32_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_uint32, opt, flag, def, comment)
#define Define_int64_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_int64, opt, flag, def, comment)
#define Define_uint64_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_uint64, opt, flag, def, comment)
#define Define_string_opt(opt, flag, def, comment) Define_ImplementerOpt(flag_string, opt, flag, def, comment)

/**@}*/

/**
 * \defgroup Multi-parameter option define macros
 * Define a option which accept parameter list, so it will be parsed.
 * @{
 */
#define Define_stringlist(opt, comment) Define_Implementer_list(flag_stringlist, opt, comment)
#define Define_boollist(opt, comment) Define_Implementer_list(flag_boollist, opt, comment)
#define Define_floatlist(opt, comment) Define_Implementer_list(flag_floatlist, opt, comment)
#define Define_doublelist(opt, comment) Define_Implementer_list(flag_doublelist, opt, comment)
#define Define_int32list(opt, comment) Define_Implementer_list(flag_int32list, opt, comment)
#define Define_uint32list(opt, comment) Define_Implementer_list(flag_uint32list, opt, comment)
#define Define_int64list(opt, comment) Define_Implementer_list(flag_int64list, opt, comment)
#define Define_uint64list(opt, comment) Define_Implementer_list(flag_uint64list, opt, comment)

/**@}*/

/**
 * \defgroup Self-define multi-parameter option define macros
 * Define a option which accept parameter list, so it will be parsed.
 */
#define Define_stringlist_opt(opt, flag, comment) Define_Implementer_listOpt(flag_stringlist, opt, flag, comment)
#define Define_int32list_opt(opt, flag, comment) Define_Implementer_listOpt(flag_int32list, opt, flag, comment)
#define Define_uint32list_opt(opt, flag, comment) Define_Implementer_listOpt(flag_uint32list, opt, flag, comment)
#define Define_int64list_opt(opt, flag, comment) Define_Implementer_listOpt(flag_int64list, opt, flag, comment)
#define Define_uint64list_opt(opt, flag, comment) Define_Implementer_listOpt(flag_uint64list, opt, flag, comment)
#define Define_floatlist_opt(opt, flag, comment) Define_Implementer_listOpt(flag_floatlist, opt, flag, comment)
#define Define_doublelist_opt(opt, flag, comment) Define_Implementer_listOpt(flag_doublelist, opt, flag, comment)

/**@}*/

Declare_bool(help);
Declare_stringlist(unknown_trash);
#endif // SIMPLE_FLAGS_H
