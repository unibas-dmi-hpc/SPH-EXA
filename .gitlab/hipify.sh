parallel -j+0 hipify-perl -inplace ::: `find . -name '*.h' -o -name '*.cuh' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu'`
