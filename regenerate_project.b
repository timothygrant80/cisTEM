libtoolize --force || glibtoolize
aclocal
autoheader --force
autoconf
automake --add-missing --copy

