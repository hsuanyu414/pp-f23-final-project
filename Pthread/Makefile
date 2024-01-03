CC=g++ 
SRC=pthread
SRC2=pthread_balancing
FLAGS= -lpthread -march=native

all:$(SRC).out $(SRC2).out

$(SRC).out: $(SRC).cpp
	$(CC) $(SRC).cpp -o $(SRC).out $(FLAGS)
$(SRC2).out: $(SRC2).cpp
	$(CC) $(SRC2).cpp -o $(SRC2).out $(FLAGS)

clean:
	rm -f *.out *.o
