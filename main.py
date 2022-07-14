from quantum import *
from gates import *
import tensorflow

def quantum_randbit():
    a = QRegister(1, '0')
    a.apply(H)
    #lmao
    return a.measure()


for i in range(32):
    print(quantum_randbit(), end='')
print()