#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang
# copy all files committed by specific users to new repo, useful when git history is messed up

outdir="/home/bhuang/transformers_new"

# cmdoutput=$(git log --no-merges --author="bhuang@zaion.ai" --author="bofenghuang7@gmail.com" --name-only --pretty=format:"" | sort -u)
cmdoutput=$(find . -type f -name "tmp*.py")

for filepath in $cmdoutput; do
    echo $filepath
    # exit 0;

    # files commited but don't exist anymore
    [ -f $filepath ] || continue

    targerfilepath=$outdir/$filepath
    targerdir=${targerfilepath%/*}
    [ -d $targerdir ] || mkdir -p $targerdir

    cp $filepath $targerfilepath
done
