#!/bin/bash

# Allow version control of vscode settings at .vscode_shared/UserName, without having defaults brought in on updates (i.e. don't track .vscode)
# Not sure bash is good for everyone, they can adjust as needed.

if [ "$#" -ne 1 ]; then
    echo "WARNING: you need to specify a users name FirstLast"
    exit 1
fi

NAME=${1}

case $NAME in
    "AlexisRohou") 
        ;;
    "BenHimes")
        ;;
    *)
        echo "WARNING, the Name $NAME is not recognized"
        exit 1
esac

cd .vscode
echo "Linking vscode settings for user $NAME"
ls ../.vscode_shared/${NAME} | while read FILE_NAME ; do
    echo $FILE_NAME
    ln -sf ../.vscode_shared/${NAME}/$FILE_NAME $(basename $FILE_NAME)
    done
