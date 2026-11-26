"""Escenario compartido por todos los módulos (módulo 1).

Datos sintéticos de sensores, partición, votación por mayoría, función de costo
(Terreno) y el registro de resultados que devuelven los algoritmos de búsqueda.
"""
import math
import random
from dataclasses import dataclass
from pathlib import Path

EPS = 1e-12
ALFA, BETA = 1.0, 0.08
# Inicio didáctico elegido para mostrar un mínimo local no global.
# No se cambian las predicciones ni las reglas de aceptación para obtenerlo.
ESTADO_INICIAL = (1, 1, 0, 0, 1, 1)
SEMILLA_DATOS = 2026
RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"


def generar_datos(semilla=SEMILLA_DATOS, cantidad=600):
    """Lecturas SINTÉTICAS de máquinas; 0=normal, 1=posible falla.

    La regla no lineal y el ruido simulan relaciones entre sensores. No son
    mediciones industriales reales ni validan un sistema de mantenimiento.
    """
    rng = random.Random(semilla)
    X, y = [], []
    for _ in range(cantidad):
        a, b, c, d = [rng.uniform(-1, 1) for _ in range(4)]
        riesgo = (a + 0.8 * b - 0.6 * c + 0.8 * a * b
                  + 1.2 * (d * d - 1 / 3) + rng.gauss(0, 0.25))
        etiqueta = int(riesgo > 0)
        if rng.random() < 0.04:
            etiqueta = 1 - etiqueta
        X.append([70 + 20 * a, 4 + 2 * b, 5 + 1.5 * c, 12 + 4 * d])
        y.append(etiqueta)
    return X, y


def dividir_datos(y, semilla=17):
    """Partición estratificada ~60/20/20, con índices disjuntos."""
    rng = random.Random(semilla)
    entrenamiento, validacion, prueba = [], [], []
    for clase in sorted(set(y)):
        indices = [i for i, etiqueta in enumerate(y) if etiqueta == clase]
        rng.shuffle(indices)
        a, b = int(0.6 * len(indices)), int(0.8 * len(indices))
        entrenamiento.extend(indices[:a])
        validacion.extend(indices[a:b])
        prueba.extend(indices[b:])
    for grupo in (entrenamiento, validacion, prueba):
        rng.shuffle(grupo)
    return entrenamiento, validacion, prueba


def validar_estado(estado, m):
    if len(estado) != m or any(bit not in (0, 1) for bit in estado) or not any(estado):
        raise ValueError(f"El estado debe tener {m} bits y al menos un modelo activo.")


def vecinos(estado):
    """Agregar o quitar UN modelo; el ensemble vacío es inválido."""
    for i in range(len(estado)):
        candidato = list(estado)
        candidato[i] = 1 - candidato[i]
        if any(candidato):
            yield tuple(candidato)


def votacion(predicciones):
    """Mayoría simple; en empate se elige la clase 0 de forma determinista."""
    if not predicciones:
        raise ValueError("No se puede votar sin modelos.")
    return [int(sum(votos) > len(predicciones) / 2) for votos in zip(*predicciones)]


def exactitud(y, predicciones):
    return sum(a == b for a, b in zip(y, predicciones)) / len(y)


def vector(estado):
    return "[" + ",".join(map(str, estado)) + "]"


class Terreno:
    """Costo sobre validación, calculado con predicciones de modelos entrenados."""

    def __init__(self, predicciones, etiquetas, alfa=ALFA, beta=BETA):
        self.predicciones, self.etiquetas = predicciones, etiquetas
        self.m, self.alfa, self.beta = len(predicciones), alfa, beta
        if not predicciones or not etiquetas or any(len(p) != len(etiquetas) for p in predicciones):
            raise ValueError("Predicciones o etiquetas vacías/incompatibles.")
        if alfa <= 0 or beta < 0 or not math.isfinite(alfa + beta):
            raise ValueError("Se requiere alfa > 0 y beta >= 0, ambos finitos.")
        self.cache = {}

    def evaluar(self, estado):
        estado = tuple(estado)
        validar_estado(estado, self.m)
        if estado not in self.cache:
            pred = votacion([p for p, activo in zip(self.predicciones, estado) if activo])
            accuracy = exactitud(self.etiquetas, pred)
            costo = self.alfa * (1 - accuracy) + self.beta * sum(estado) / self.m
            self.cache[estado] = (costo, accuracy)
        return self.cache[estado]


@dataclass
class Resultado:
    algoritmo: str
    inicial: tuple
    estado: tuple
    costo: float
    accuracy: float
    iteraciones: int
    movimientos: int
    evaluaciones: int
    combinaciones_unicas: int
    traza: list
    ultimo_estado: tuple
    ultimo_costo: float
    peores_aceptados: int = 0
    semilla: int | None = None
