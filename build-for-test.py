# -*- coding: utf-8 -*-

import os
import sys
import urllib.request
import platform
import multiprocessing

def PrepareLinuxLuaDep():
    lua_version = '5.3.6'
    lua_package = '/tmp/lua-' + lua_version + '.tar.gz'
    if not os.path.exists(lua_package):
        urllib.request.urlretrieve('https://www.lua.org/ftp/lua-' + lua_version + '.tar.gz', lua_package)
    lua_dir = '/tmp/lua-' + lua_version
    if not os.path.isdir(lua_dir):
        os.system('cd /tmp; tar -xf ' + lua_package)
        os.system('cd ' + lua_dir + '; make posix -j8 MYCFLAGS="-DLUA_USE_DLOPEN -fPIC" MYLIBS=-ldl')
    return lua_dir

def GenericBuild(build_type, build_dir, options):
    os.makedirs(build_dir, exist_ok = True)
    os.chdir(build_dir)
    cpu_count = multiprocessing.cpu_count()
    cmd = 'cmake ' + options + ' -DCMAKE_BUILD_TYPE=' + build_type + ' -DCMAKE_INSTALL_PREFIX=install .. && cmake --build . -j ' + str(cpu_count) + ' --config ' + build_type + ' && cmake --build . --target install -j ' + str(cpu_count) + ' --config ' + build_type
    print('cmd -> ' + cmd, flush = True)
    os.system(cmd)

def BuildCuda(build_type, options):
    GenericBuild(build_type, 'cuda-build', options + ' -DHPCC_USE_CUDA=ON')

def BuildX64(build_type, options):
    GenericBuild(build_type, 'x86-64-build', options + ' -DHPCC_USE_X86_64=ON')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' [x86_64 | cuda]')
        sys.exit(-1)

    build_type = 'Release'
    options = '-DPPLNN_ENABLE_PYTHON_API=ON'
    os_type = platform.system()

    if os_type == 'Linux':
        lua_dir = PrepareLinuxLuaDep()
        options = options + ' -DPPLNN_ENABLE_LUA_API=ON -DLUA_INCLUDE_DIR=' + lua_dir + '/src -DLUA_LIBRARIES=' + lua_dir + '/src/liblua.a'
    elif os_type == 'Windows':
        options = options + ' -G "Visual Studio 14 2015 Win64"'

    if sys.argv[1] == 'x86_64':
        BuildX64(build_type, options)
    elif sys.argv[1] == 'cuda':
        BuildCuda(build_type, options)
    else:
        print('ERROR! unsupported arch[' + sys.argv[1] + ']')
        sys.exit(-1)
