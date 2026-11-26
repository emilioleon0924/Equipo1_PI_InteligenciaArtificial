"""Módulos 2 y 3: Hill Climbing y Simulated Annealing."""
import math
import random

from escenario import EPS, Resultado, vecinos, vector


# MÓDULO 2: BÚSQUEDA LOCAL
def hill_climbing(terreno, inicial, verbose=True):
    actual = tuple(inicial)
    costo, accuracy = terreno.evaluar(actual)
    consultados, evaluaciones, movimientos = {actual}, 1, 0
    traza, iteraciones = [], 0
    if verbose:
        print(f"\nMÓDULO 2 · HILL CLIMBING\nInicio: {vector(actual)} | costo={costo:.6f}")
    while True:
        iteraciones += 1
        candidatos = []
        for vecino in vecinos(actual):
            c, acc = terreno.evaluar(vecino)
            evaluaciones += 1
            consultados.add(vecino)
            candidatos.append((c, vecino, acc))
            if verbose:
                print(f"  Ronda {iteraciones}: vecino {vector(vecino)}, costo={c:.6f}")
        if not candidatos:
            break
        # Desempate reproducible por orden lexicográfico del vector binario.
        c, vecino, acc = min(candidatos)
        if c >= costo - EPS:
            break  # Condición estricta: no aceptar igualdad ni empeoramiento.
        traza.append({"ronda": iteraciones, "desde": actual, "hasta": vecino,
                    "costo_desde": costo, "costo_hasta": c})
        actual, costo, accuracy = vecino, c, acc
        movimientos += 1
        if verbose:
            print(f"  MEJORA -> {vector(actual)} | costo={costo:.6f}")
    if verbose:
        print(f"ESTANCAMIENTO: {vector(actual)} | costo={costo:.6f}")
        print("Ningún vecino válido de un bit reduce el costo.")
        print(f"Rondas: {iteraciones} (incluye comprobación final); movimientos: {movimientos}.")
    return Resultado("Hill Climbing", tuple(inicial), actual, costo, accuracy,
                    iteraciones, movimientos, evaluaciones, len(consultados),
                    traza, actual, costo)


def probabilidad_aceptacion(delta_f, temperatura):
    if temperatura <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    # deltaF = costo_actual - costo_vecino: negativo para una solución peor.
    # Equivale a Math.exp(deltaF / temperatura) en Java/JavaScript.
    return 1.0 if delta_f >= 0 else math.exp(delta_f / temperatura)


# MÓDULO 3: BÚSQUEDA ESTOCÁSTICA
def simulated_annealing(terreno, inicial, semilla=7, temperatura_inicial=0.08,
                        enfriamiento=0.99, temperatura_minima=0.0001,
                        max_iteraciones=1000, verbose=True, detallado=False):
    if (not 0 < temperatura_minima < temperatura_inicial
            or not 0 < enfriamiento < 1 or max_iteraciones < 1
            or not math.isfinite(temperatura_inicial)):
        raise ValueError("Temperaturas, enfriamiento o límite de iteraciones inválidos.")
    rng = random.Random(semilla)
    actual = tuple(inicial)
    costo, accuracy = terreno.evaluar(actual)
    mejor, mejor_costo, mejor_accuracy = actual, costo, accuracy
    temperatura = temperatura_inicial
    consultados, evaluaciones = {actual}, 1
    traza, movimientos, peores, iteracion = [], 0, 0, 0
    if verbose:
        print(f"\nMÓDULO 3 · SIMULATED ANNEALING\nInicio: {vector(actual)} | costo={costo:.6f}")
        print(f"T0={temperatura}; enfriamiento={enfriamiento}; semilla={semilla}")
    while temperatura > temperatura_minima and iteracion < max_iteraciones:
        # Se propone un bit uniforme entre TODOS los bits. Quitar el único modelo
        # produce una propuesta inválida que se rechaza: conserva la simetría.
        iteracion += 1
        candidato = list(actual)
        j = rng.randrange(terreno.m)
        candidato[j] = 1 - candidato[j]
        candidato = tuple(candidato)
        desde, costo_desde = actual, costo
        valido = any(candidato)
        u = rng.random()
        c, delta, p, aceptado, nuevo_mejor = None, None, 0.0, False, False
        if valido:
            c, acc = terreno.evaluar(candidato)
            evaluaciones += 1
            consultados.add(candidato)
            delta = costo - c
            p = probabilidad_aceptacion(delta, temperatura)
            aceptado = u < p
            if aceptado:
                actual, costo, accuracy = candidato, c, acc
                movimientos += 1
                peores += int(delta < -EPS)
                if costo < mejor_costo - EPS:
                    mejor, mejor_costo, mejor_accuracy = actual, costo, accuracy
                    nuevo_mejor = True
        paso = {"iteracion": iteracion, "temperatura": temperatura,
                "desde": desde, "costo_desde": costo_desde, "candidato": candidato,
                "valido": valido, "costo_candidato": c, "delta_f": delta,
                "probabilidad": p, "aleatorio": u, "aceptado": aceptado,
                "actual": actual, "costo_actual": costo, "mejor_costo": mejor_costo}
        traza.append(paso)
        if verbose and (detallado or iteracion <= 10 or nuevo_mejor
                        or (aceptado and delta < -EPS) or iteracion % 100 == 0):
            accion = ("RECHAZA VACÍO" if not valido else
                      "ACEPTA EMPEORAMIENTO" if aceptado and delta < -EPS else
                      "ACEPTA" if aceptado else "RECHAZA")
            print(f"  i={iteracion:3d} T={temperatura:.6f} {vector(desde)} -> {vector(candidato)}")
            print(f"    {accion}; p={p:.6f}, u={u:.6f}; actual={costo:.6f}; mejor={mejor_costo:.6f}"
                  + ("  NUEVO MEJOR" if nuevo_mejor else ""))
        temperatura *= enfriamiento
    if verbose:
        print(f"MEJOR ENCONTRADO: {vector(mejor)} | costo={mejor_costo:.6f}")
        print(f"Último estado visitado: {vector(actual)} | costo={costo:.6f}")
        print(f"Iteraciones: {iteracion}; empeoramientos aceptados: {peores}.")
        print("El módulo 4 verificará si el mejor encontrado es óptimo global.")
    # Se conserva el mejor visitado: el último estado puede ser peor.
    return Resultado("Simulated Annealing", tuple(inicial), mejor, mejor_costo, mejor_accuracy,
                     iteracion, movimientos, evaluaciones, len(consultados), traza,
                     actual, costo, peores, semilla)
