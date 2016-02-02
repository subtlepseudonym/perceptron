# Perceptron
Various Perceptron Related Code

current use for perceptron.py: <br />
python perceptron.py LEARN\_RATE(=0.01) DATA\_DIM(=2) NUM\_TRAIN\_CASES(=40) NUM\_TEST\_CASES(=120) <br />
perceptron.py will also accepts no arguments in which case the default values (=default\_val) will be used

perceptron.py uses a simple step function as its activation function and a learning function of the form: <br />
w(t+1) = w(t) + LEARN\_RATE * (EXPECTED\_OUTPUT - ACTUAL\_OUTPUT) * INPUT
