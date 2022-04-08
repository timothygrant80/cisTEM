# Build the simple module example

# The module interace must be build before the module implementation.
# In this case the impl is in the interface.
# In cisTEMx module interface files are module_name.cpp and implmenetation (if sep) module_name_impl.cpp

INTERFACES := hello.cpp hello2.cpp
MODULE_OBJECTS := $(INTERFACES:.cpp=.o)

SRCS := simple_module.cpp
OBJECTS := $(SRCS:.cpp=.o)

all: module_app

modules.a : $(INTERFACES)
	g++ -D_CISTEM_MODULES -c -fmodules-ts -std=c++20 $^ 
	ar rvs -o $@ $(MODULE_OBJECTS)

$(OBJECTS) : $(SRCS)
	g++ -D_CISTEM_MODULES -c -fmodules-ts -std=c++20 $^ 


module_app :  $(OBJECTS) modules.a
	g++ -fmodules-ts -std=c++20 $^ -o $@

clean:
	rm -f $(OBJECTS) *.o modules.a module_app