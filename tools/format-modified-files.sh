#!/bin/bash

formatter="/usr/bin/clang-format-12"
if ! [ -f "$formatter" ]; then
    echo "cannot find '$formatter'" >&2
    exit -1
fi

filelist=`git diff --name-only master`
for entry in ${filelist[@]}; do
    if ! [ -f "$entry" ]; then # deleted file
        echo "skip [$entry]"
        continue
    fi
    if [[ "$entry" =~ "generated" ]]; then
        echo "skip [$entry]"
        continue
    fi

    filetype=${entry##*.}
    if [[ "$filetype" == "cpp" || "$filetype" == "cc" || "$filetype" == "h" || "$filetype" == "hpp" ]]; then
        echo -n "formatting [$entry]..."
        clang-format-12 -i $entry
        if [ $? -eq 0 ]; then
            echo " done"
        else
            exit -1
        fi
    fi
done
