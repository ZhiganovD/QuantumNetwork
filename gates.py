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