md pplnn-build
cd pplnn-build
cmake %* -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . -j --config Release
cmake --build . --target install -j --config Release
cd ..
