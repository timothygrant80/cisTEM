rm -fr m4
mkdir m4
libtoolize || glibtoolize
aclocal
autoreconf
automake --add-missing --copy
