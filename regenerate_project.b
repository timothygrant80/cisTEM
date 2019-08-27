rm -fr m4
mkdir m4
cd m4
ln -s ../ax_cuda.m4 ax_cuda.m4
cd ..
libtoolize || glibtoolize
aclocal
autoconf
automake --add-missing --copy




