cpp_compiler = g++
c_complier = gcc

# variables
exec_file = admm_learner
objs = test.o
obj2 = io.o
# auto compiler
$(exec_file) : $(objs) $(obj2)
	$(cpp_compiler) -o $(exec_file) $(objs) $(obj2)

$(obj2) : io/io.cc io.h
	$(cpp_compiler) -c io/io.cc

$(objs) : test.cc io.h base.h
	$(cpp_compiler) -c test.cc


# delete middle and exec file (.PHONY represent clean is pseudo-object file)
.PHONY : clean
clean :
	-rm -f $(exec_file) $(objs) $(obj2)
