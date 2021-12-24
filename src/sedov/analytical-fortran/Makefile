PROCESS	= analytical.run
##################################################################################################################################33

all: 
	make clean
	gfortran analytical.f -o $(PROCESS)
	./$(PROCESS)
	#sh ./analytical.sh

clean:
	rm -f *.run *.o *.dat *.png
