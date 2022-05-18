#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PPLNN_DIR=${SCRIPT_DIR}/../..

PACKAGE_DIR=/tmp/pyppl-package

rm -rf ${PACKAGE_DIR} # remove old packages
cp -r "${SCRIPT_DIR}" ${PACKAGE_DIR}
cp ${PPLNN_DIR}/VERSION ${PACKAGE_DIR}
cp -r ${PPLNN_DIR}/pplnn-build/install/lib/pyppl ${PACKAGE_DIR}
cd ${PACKAGE_DIR}
python3 setup.py bdist_wheel
if [ $? -eq 0 ]; then
    echo '------------------------------'
    echo "building finished. wheel package is in \`${PACKAGE_DIR}/dist\`."
fi
