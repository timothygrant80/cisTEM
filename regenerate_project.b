rm -fr m4
mkdir m4
cd m4
ln -s ../ax_cuda.m4 ax_cuda.m4
ln -s ../additional_programs.m4 additional_programs.m4
cd ..
libtoolize --force || glibtoolize
aclocal
autoheader --force
autoconf
automake --add-missing --copy

