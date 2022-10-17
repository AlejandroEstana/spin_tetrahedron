# spin_tetrahedron

# To compile the code we need first to install TACO from here:

	https://github.com/tensor-compiler/taco

# Then link to it. For example: 

	g++ -std=c++11 -O3 -DNDEBUG -DTACO -I../taco/include -L../taco/build/lib main.cpp -o main -ltaco	
