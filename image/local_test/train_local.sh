#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/* || true
rm test_dir/output/* || true

docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
