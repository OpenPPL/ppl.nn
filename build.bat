md pplnn-build
cd pplnn-build
cmake %* -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install ..
cmake --build . --config Release
cmake --install . --prefix %cd%/install
cd ..
