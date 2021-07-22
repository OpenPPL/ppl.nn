#!/bin/bash

echo "usage example: [FORMATTER='/path/to/formatter --option foo'] ./tools/format-modified-files.sh"
echo '---------------------------------------'

if [ -z "$FORMATTER" ]; then
    default_formatter='/usr/bin/clang-format-12'
    if ! [ -f ${default_formatter} ]; then
        echo "FORMATTER is not set." >&2
        exit -1
    fi
    formatter="${default_formatter} -i"
else
    formatter=$FORMATTER
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
        echo -n "$formatter [$entry]..."
        $formatter $entry
        if [ $? -eq 0 ]; then
            echo " done"
        else
            exit -1
        fi
    else
        echo "skip [$entry]"
    fi
done
