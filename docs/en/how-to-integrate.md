There are usually 4 ways to integrate `ppl.nn` into your applications.

## 1. Using Cmake `find_package` (recommended)

After building and installing `ppl.nn`, there is a directory(specified by `CMAKE_INSTALL_PREFIX`) containing header files, libraries and cmake configurations. The entry point for finding `ppl.nn` is `${CMAKE_INSTALL_PREFIX}/lib/cmake/ppl/pplnn-config.cmake`. Here is a sample snippet of `CMakeLists.txt`:

```cmake
set(pplnn_DIR "<pplnn_install_dir>/lib/cmake/ppl")
find_package(pplnn REQUIRED)

target_include_directories(<target> PUBLIC ${PPLNN_INCLUDE_DIRS})
target_link_libraries(<target> PUBLIC ${PPLNN_LIBRARIES})
```

Refer to `${CMAKE_INSTALL_PREFIX}/lib/cmake/ppl/pplnn-config.cmake` for other cmake variables like versions and supported devices. In this form you can only use public APIs defined in `${CMAKE_INSTALL_PREFIX}/include/ppl`.

A simple example [integration](../../samples/cpp/integration) shows how to use `pplnn-config.cmake` to integrate x86 engine into an application.

[integration-cuda](../../samples/cpp/integration-cuda) is a CUDA example.

## 2. Using `ppl.nn` as a Source Dependency

You can also import `ppl.nn`'s source like this:

```cmake
add_subdirectory(pplnn_source_dir)

target_link_libraries(<target> PUBLIC pplnn_static)
target_include_directories(<target> PUBLIC <pplnn_source_dir>/include <pplnn_source_dir>/src)
```

Both public and internal APIs are avaliable in this form. **NOTE** internal APIs are changed frequently and not guaranteed to keep compatibility.

## 3. Embedding Your Source Code into `ppl.nn`

`ppl.nn` provides the following variables:

* `PPLNN_SOURCE_EXTERNAL_SOURCES`: source files built and packed into `ppl.nn`
* `PPLNN_SOURCE_EXTERNAL_INCLUDE_DIRECTORIES`: extra include directories when building your code
* `PPLNN_SOURCE_EXTERNAL_LINK_DIRECTORIES`: extra link directories when building your code
* `PPLNN_SOURCE_EXTERNAL_LINK_LIBRARIES`: extra link libraries when building your code
* `PPLNN_SOURCE_EXTERNAL_COMPILE_DEFINITIONS`: extra compile definitions when building your code
* `PPLNN_BINARY_EXTERNAL_LINK_LIBRARIES`: extra link libraries needed by your code when linking `ppl.nn`
* `PPLNN_BINARY_EXTERNAL_COMPILE_DEFINITIONS`: extra compile definitions needed by your code when linking `ppl.nn`

In this form you don't need to write extra configurations and `.o`s are packed into the `ppl.nn` libraries.

You may also set `PPLNN_VERSION_STR`(in the form of `<major>.<minor>.<patch>`) and `PPLNN_COMMIT_STR`(a string) to the internal version info to override the default values.

Example:

```cmake
set(PPLNN_SOURCE_EXTERNAL_SOURCES foo.cc bar.cc)
set(PPLNN_SOURCE_EXTERNAL_INCLUDE_DIRECTORIES /path/to/include)
set(PPLNN_BINARY_EXTERNAL_LINK_LIBRARIES baz_static)

set(PPLNN_VERSION_STR '1.0.0')
set(PPLNN_COMMIT_STR '1e1065')

add_subdirectory(pplnn_source_dir)
```

## 4. Manually Specifying Everything

Good luck!
