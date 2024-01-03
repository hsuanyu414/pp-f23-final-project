CC=g++ 
SRC=pthread
FLAGS= -lpthread -march=native

all:$(SRC).out

$(SRC).out: $(SRC).cpp
	$(CC) $(SRC).cpp -o $(SRC).out $(FLAGS)

clean:
	rm -f *.out *.o
