"""MÓDULO 10: problema de satisfacción de restricciones (CSP).

El ensemble se modela como CSP:
  Variables:    X1..X6, una por modelo (Xi = 1 si el modelo Mi está activo).
  Dominios:     Di = {0, 1}.
  Restricciones:
    R1 (unaria, una por variable): Xi = 1  =>  accuracy de validación de Mi > 80 %.
        Un modelo que no supera el umbral mínimo de calidad no puede votar.
    R2 (binaria, X1 y X5): X1 + X5 <= 1.
        Regresión logística y SVM lineal son ambos modelos LINEALES; activarlos
        juntos duplica la misma frontera de decisión y no aporta diversidad.
    R3 (global, X1..X6): X1 + ... + X6 ∈ {3, 5}.
        Un ensemble de votación necesita al menos 3 modelos y un número impar
        de votos para que la mayoría nunca empate.
"""
from dataclasses import dataclass
from typing import Callable

UMBRAL_ACCURACY = 0.80
TAMANOS_VALIDOS = (3, 5)
EPS = 1e-12


@dataclass
class Restriccion:
    nombre: str
    tipo: str             # "unaria", "binaria" o "global"
    alcance: tuple        # índices (0..5) de las variables que involucra
    descripcion: str
    revisar: Callable     # asignación completa/parcial -> (cumple, detalle)


class CSPEnsemble:
    def __init__(self, modelos, accuracies):
        """modelos: [(id, nombre, objeto)]; accuracies: accuracy de validación de cada modelo."""
        self.ids = [identificador for identificador, _, _ in modelos]
        self.nombres = [nombre for _, nombre, _ in modelos]
        self.accuracies = list(accuracies)
        self.variables = [f"X{i + 1}" for i in range(len(modelos))]
        self.dominios = {v: (0, 1) for v in self.variables}
        self.restricciones = self._crear_restricciones()

    def _crear_restricciones(self):
        restricciones = []
        # R1 · UNARIAS: una restricción por variable, solo depende de Xi.
        for i, acc in enumerate(self.accuracies):
            def revisar(a, i=i, acc=acc):
                cumple = a[i] == 0 or acc > UMBRAL_ACCURACY + EPS
                return cumple, f"{self.ids[i]} activo con accuracy {acc:.2%} (debe ser > 80%)"
            restricciones.append(Restriccion(
                f"R1.{i + 1}", "unaria", (i,),
                f"X{i + 1} = 1 => acc({self.ids[i]}) = {acc:.2%} > 80%", revisar))

        # R2 · BINARIA: X1 (regresión logística) y X5 (SVM lineal).
        def revisar_binaria(a):
            return a[0] + a[4] <= 1, "M1 y M5 activos a la vez (dos modelos lineales)"
        restricciones.append(Restriccion(
            "R2", "binaria", (0, 4), "X1 + X5 <= 1 (no dos modelos lineales)", revisar_binaria))

        # R3 · GLOBAL: involucra a las seis variables.
        def revisar_global(a):
            activos = sum(a)
            return activos in TAMANOS_VALIDOS, (f"{activos} modelo(s) activo(s); "
                                                f"se requieren 3 o 5")
        restricciones.append(Restriccion(
            "R3", "global", tuple(range(len(self.variables))),
            "X1 + ... + X6 ∈ {3, 5} (mayoría sin empates)", revisar_global))
        return restricciones

    def verificar(self, asignacion):
        """Revisa TODAS las restricciones; devuelve (factible, lista de incumplidas)."""
        incumplidas = []
        for r in self.restricciones:
            cumple, detalle = r.revisar(asignacion)
            if not cumple:
                incumplidas.append((r, detalle))
        return not incumplidas, incumplidas

    def backtracking(self):
        """Búsqueda con retroceso: asigna X1..X6 en orden y poda en cuanto una
        restricción con todas sus variables asignadas se incumple."""
        soluciones, nodos = [], 0

        def consistente(parcial):
            k = len(parcial)
            return all(r.revisar(parcial)[0] for r in self.restricciones
                       if max(r.alcance) == k - 1)   # recién quedó completa

        def asignar(parcial):
            nonlocal nodos
            if len(parcial) == len(self.variables):
                soluciones.append(tuple(parcial))
                return
            for valor in self.dominios[self.variables[len(parcial)]]:
                nodos += 1
                parcial.append(valor)
                if consistente(parcial):
                    asignar(parcial)
                parcial.pop()

        asignar([])
        return soluciones, nodos


def vector(estado):
    return "[" + ",".join(map(str, estado)) + "]"


def ejecutar_modulo_10(csp, terreno, resultados):
    """resultados: lista de objetos Resultado (HC, SA y AG)."""
    print("\nMÓDULO 10 · PROBLEMA DE SATISFACCIÓN DE RESTRICCIONES (CSP)")
    print("Variables y dominios:")
    for v, i, n, acc in zip(csp.variables, csp.ids, csp.nombres, csp.accuracies):
        print(f"  {v} ∈ {{0,1}}  ->  {i} · {n:<29} (accuracy validación {acc:.2%})")
    print("Restricciones:")
    for r in csp.restricciones:
        alcance = ",".join(csp.variables[j] for j in r.alcance)
        print(f"  {r.nombre:<5} [{r.tipo:<7}] sobre ({alcance}): {r.descripcion}")

    print("\nVerificación de la mejor solución de cada método:")
    for res in resultados:
        factible, incumplidas = csp.verificar(res.estado)
        print(f"  {res.algoritmo:<20} {vector(res.estado)} costo={res.costo:.6f} -> "
              f"{'FACTIBLE' if factible else 'NO FACTIBLE'}")
        for r, detalle in incumplidas:
            print(f"      incumple {r.nombre} ({r.tipo}): {detalle}")

    soluciones, nodos = csp.backtracking()
    print(f"\nBacktracking: {nodos} nodos explorados; {len(soluciones)} asignaciones factibles "
          f"de 64 posibles.")
    ordenadas = sorted(soluciones, key=lambda e: (terreno.evaluar(e)[0], e))
    for e in ordenadas:
        costo, acc = terreno.evaluar(e)
        print(f"  {vector(e)} | modelos={sum(e)} | costo={costo:.6f} | acc.val={acc:.2%}")
    if ordenadas:
        mejor = ordenadas[0]
        costo, acc = terreno.evaluar(mejor)
        print(f"MEJOR SOLUCIÓN FACTIBLE: {vector(mejor)} | costo={costo:.6f} | acc.val={acc:.2%}")
    return ordenadas
