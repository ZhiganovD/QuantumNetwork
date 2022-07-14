from quantum import *

I = QGate([[1, 0], [0, 1]])
H = QGate(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
X = QGate([[0, 1], [1, 0]])
Y = QGate([[0, -1j], [1j, 0]])
Z = QGate([[1, 0], [0, -1]])
T = QGate([[0, 0], [0, (1 + 1j) / np.sqrt(2)]])
def ry(state, phi):
    return np.array([
        [np.cos(phi / 2), -np.sin(phi / 2)],
        [np.sin(phi / 2),  np.cos(phi / 2)]
    ]) @ state

def rx(state, phi):
    return np.array([
        [np.cos(phi / 2), -1j * np.sin(phi / 2)],
        [-1j * np.sin(phi / 2),  np.cos(phi / 2)]
    ]) @ state

def rz(state, phi):
    return np.array([
        [np.e ** (-0.5 * 1j * phi), 0],
        [0,  np.e ** (0.5 * 1j * phi)]
    ]) @ state

def u1(state, phi):
    return np.array([[1, 0], [0, np.exp(1j * phi)]]) @ state

CNOT = QGate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])