import math
import random
from collections import Counter


def sigmoide(z):
    """Sigmoide estable: evita desbordamientos de exp."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def producto(a, b):
    return sum(x * y for x, y in zip(a, b))


class Estandarizador:
    def fit(self, X):
        self.medias = [sum(c) / len(X) for c in zip(*X)]
        self.escalas = [
            math.sqrt(sum((r[j] - m) ** 2 for r in X) / len(X)) or 1.0
            for j, m in enumerate(self.medias)
        ]
        return self

    def transform(self, X):
        return [
            [(v - m) / s for v, m, s in zip(r, self.medias, self.escalas)]
            for r in X
        ]


class RegresionLogistica:
    """Descenso de gradiente sobre entropía cruzada + regularización L2."""

    def fit(self, X, y):
        self.w, self.b = [0.0] * len(X[0]), 0.0
        for _ in range(400):
            dw, db = [0.0] * len(self.w), 0.0
            for x, etiqueta in zip(X, y):
                error = sigmoide(producto(self.w, x) + self.b) - etiqueta
                db += error
                for j, valor in enumerate(x):
                    dw[j] += error * valor
            for j in range(len(self.w)):
                self.w[j] -= 0.12 * (dw[j] / len(X) + 0.002 * self.w[j])
            self.b -= 0.12 * db / len(X)
        return self

    def predict(self, X):
        return [int(sigmoide(producto(self.w, x) + self.b) >= 0.5) for x in X]


class ArbolDecision:
    """CART binario con impureza Gini, profundidad y tamaño de hoja limitados."""

    def __init__(self, profundidad=4, hoja_minima=6, max_features=None, semilla=42):
        self.profundidad = profundidad
        self.hoja_minima = hoja_minima
        self.max_features = max_features
        self.semilla = semilla

    def fit(self, X, y):
        rng = random.Random(self.semilla)

        def construir(indices, nivel):
            total = len(indices)
            positivos = sum(y[i] for i in indices)
            # En cualquier empate de clase se elige 0, también en el ensemble.
            mayoria = int(positivos > total / 2)
            if (nivel >= self.profundidad or positivos in (0, total)
                    or total < 2 * self.hoja_minima):
                return mayoria
            atributos = list(range(len(X[0])))
            if self.max_features is not None:
                atributos = rng.sample(atributos, self.max_features)
            mejor_impureza = 2 * positivos * (total - positivos) / total
            corte = None
            for j in atributos:
                orden = sorted(indices, key=lambda i: X[i][j])
                positivos_izq = 0
                for k in range(1, total):
                    positivos_izq += y[orden[k - 1]]
                    if k < self.hoja_minima or total - k < self.hoja_minima:
                        continue
                    if X[orden[k - 1]][j] == X[orden[k]][j]:
                        continue
                    positivos_der = positivos - positivos_izq
                    impureza = (2 * positivos_izq * (k - positivos_izq) / k
                                + 2 * positivos_der * (total - k - positivos_der)
                                / (total - k))
                    if impureza < mejor_impureza - 1e-12:
                        mejor_impureza = impureza
                        umbral = (X[orden[k - 1]][j] + X[orden[k]][j]) / 2
                        corte = (j, umbral, orden[:k], orden[k:])
            if corte is None:
                return mayoria
            j, umbral, izq, der = corte
            return (j, umbral, construir(izq, nivel + 1), construir(der, nivel + 1))

        self.raiz = construir(list(range(len(X))), 0)
        return self

    def predict(self, X):
        predicciones = []
        for x in X:
            nodo = self.raiz
            while isinstance(nodo, tuple):
                j, umbral, izq, der = nodo
                nodo = izq if x[j] <= umbral else der
            predicciones.append(nodo)
        return predicciones


class RandomForest:
    """Bootstrap de muestras + subconjunto aleatorio de atributos por nodo."""

    def __init__(self, arboles=17, semilla=43):
        self.cantidad = arboles
        self.semilla = semilla

    def fit(self, X, y):
        rng = random.Random(self.semilla)
        self.arboles = []
        for _ in range(self.cantidad):
            indices = [rng.randrange(len(X)) for _ in X]
            arbol = ArbolDecision(
                profundidad=5, hoja_minima=4,
                max_features=max(1, int(math.sqrt(len(X[0])))),
                semilla=rng.randrange(1_000_000),
            )
            arbol.fit([X[i] for i in indices], [y[i] for i in indices])
            self.arboles.append(arbol)
        return self

    def predict(self, X):
        votos = [a.predict(X) for a in self.arboles]
        return [int(sum(columna) > len(votos) / 2) for columna in zip(*votos)]


class KNN:
    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict(self, X):
        salida = []
        for x in X:
            distancias = sorted(
                (sum((a - b) ** 2 for a, b in zip(x, fila)), i)
                for i, fila in enumerate(self.X)
            )
            etiquetas = [self.y[i] for _, i in distancias[:self.k]]
            conteo = Counter(etiquetas)
            salida.append(int(conteo[1] > conteo[0]))
        return salida


class SVMLineal:
    """SVM lineal primal: subgradiente de hinge loss + L2 (sin kernel)."""

    def fit(self, X, y):
        self.w, self.b = [0.0] * len(X[0]), 0.0
        for epoca in range(350):
            dw, db = [0.0] * len(self.w), 0.0
            for x, etiqueta in zip(X, y):
                signo = 2 * etiqueta - 1
                if signo * (producto(self.w, x) + self.b) < 1.0:
                    db -= signo
                    for j, valor in enumerate(x):
                        dw[j] -= signo * valor
            tasa = 0.08 / (1 + epoca / 100)
            for j in range(len(self.w)):
                self.w[j] -= tasa * (dw[j] / len(X) + 0.01 * self.w[j])
            self.b -= tasa * db / len(X)
        return self

    def predict(self, X):
        return [int(producto(self.w, x) + self.b >= 0) for x in X]


class RedNeuronal:
    """MLP 4-6-1: tanh, salida sigmoide y retropropagación con SGD."""

    def __init__(self, ocultas=6, semilla=44):
        self.ocultas, self.semilla = ocultas, semilla

    def fit(self, X, y):
        rng = random.Random(self.semilla)
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in X[0]] for _ in range(self.ocultas)]
        self.b1 = [0.0] * self.ocultas
        self.w2 = [rng.uniform(-0.5, 0.5) for _ in range(self.ocultas)]
        self.b2 = 0.0
        indices = list(range(len(X)))
        for epoca in range(180):
            rng.shuffle(indices)
            tasa = 0.025 / (1 + epoca / 150)
            for i in indices:
                x = X[i]
                h = self._oculta(x)
                error = sigmoide(producto(self.w2, h) + self.b2) - y[i]
                # Gradientes de la capa oculta con los pesos aún sin actualizar.
                errores_h = [error * w * (1 - v * v) for w, v in zip(self.w2, h)]
                for j in range(self.ocultas):
                    self.w2[j] -= tasa * (error * h[j] + 0.001 * self.w2[j])
                    self.b1[j] -= tasa * errores_h[j]
                    for k in range(len(x)):
                        self.w1[j][k] -= tasa * (errores_h[j] * x[k]
                                                + 0.001 * self.w1[j][k])
                self.b2 -= tasa * error
        return self

    def _oculta(self, x):
        return [math.tanh(producto(w, x) + b) for w, b in zip(self.w1, self.b1)]

    def predict(self, X):
        return [int(sigmoide(producto(self.w2, self._oculta(x)) + self.b2) >= 0.5)
                for x in X]


def crear_modelos():
    return [
        ("M1", "Regresión Logística", RegresionLogistica()),
        ("M2", "Árbol de Decisión", ArbolDecision()),
        ("M3", "Random Forest", RandomForest()),
        ("M4", "K-Nearest Neighbors", KNN()),
        ("M5", "Support Vector Machine lineal", SVMLineal()),
        ("M6", "Red neuronal pequeña", RedNeuronal()),
    ]
