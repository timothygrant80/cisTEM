rm -fr m4
mkdir m4
aclocal
autoconf
libtoolize || glibtoolize
automake --add-missing --copy


