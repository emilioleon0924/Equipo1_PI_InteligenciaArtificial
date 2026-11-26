"""Módulos 5 a 8: algoritmo genético para seleccionar los modelos del ensemble.

Cromosoma: los mismos 6 bits del ensemble (gen i = modelo Mi activo/inactivo).
Aptitud: se MINIMIZA el mismo costo del terreno, para comparar en igualdad.
"""
import random
import statistics

from escenario import EPS, Resultado, vector

# Configuración del algoritmo genético (módulos 5-8).
N_AG, TORNEO_AG, PROB_CRUCE_AG, PROB_MUTACION_AG, SEMILLA_AG = 10, 3, 0.9, 0.1, 38


def reparar(cromosoma, rng):
    """Un hijo vacío no es un ensemble válido: se activa un modelo al azar."""
    cromosoma = list(cromosoma)
    if not any(cromosoma):
        cromosoma[rng.randrange(len(cromosoma))] = 1
    return tuple(cromosoma)


# MÓDULO 5: INICIALIZACIÓN DE LA POBLACIÓN
def poblacion_inicial(terreno, n, rng):
    poblacion = [reparar([rng.randint(0, 1) for _ in range(terreno.m)], rng) for _ in range(n)]
    return poblacion


# MÓDULO 6: SELECCIÓN DE PADRES (TORNEO)
def seleccion_torneo(poblacion, terreno, rng, k=3):
    """Se toman k individuos al azar y gana el de menor costo."""
    competidores = rng.sample(poblacion, k)
    return min(competidores, key=lambda e: (terreno.evaluar(e)[0], e))


# MÓDULO 7: OPERADORES GENÉTICOS
def cruce_un_punto(padre1, padre2, rng):
    punto = rng.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return reparar(hijo1, rng), reparar(hijo2, rng), punto


def mutacion(cromosoma, rng, prob=0.1):
    """Cada gen se invierte con baja probabilidad."""
    nuevo = [1 - g if rng.random() < prob else g for g in cromosoma]
    return reparar(nuevo, rng)


# MÓDULO 8: CICLO GENERACIONAL Y REEMPLAZO
def algoritmo_genetico(terreno, n=10, generaciones=30, prob_cruce=0.9, prob_mutacion=0.1,
                    hijos_por_generacion=4, torneo=3, semilla=38, verbose=True):
    rng = random.Random(semilla)
    poblacion = poblacion_inicial(terreno, n, rng)
    evaluaciones = len(poblacion)
    consultados = set(poblacion)
    costo_de = lambda e: terreno.evaluar(e)[0]
    traza = []
    if verbose:
        print("\nMÓDULO 8 · CICLO GENERACIONAL Y REEMPLAZO (algoritmo genético completo)")
        print(f"Población inicial (módulo 5): N={n} individuos aleatorios; semilla={semilla}.")
        for i, ind in enumerate(poblacion, 1):
            c, acc = terreno.evaluar(ind)
            print(f"  Ind {i:2d}: {vector(ind)} | costo={c:.6f} | acc={acc:.2%}")
    mejor = min(poblacion, key=lambda e: (costo_de(e), e))
    promedio = statistics.mean(costo_de(e) for e in poblacion)
    traza.append({"generacion": 0, "mejor": mejor, "mejor_costo": costo_de(mejor), "promedio": promedio})
    if verbose:
        print(f"MEJOR DE LA GENERACIÓN 0: {vector(mejor)} | costo={costo_de(mejor):.6f} "
            f"| promedio poblacional={promedio:.6f}")
        print(f"\nCada generación: selección por torneo k={torneo} (módulo 6), cruce de un punto "
            f"p={prob_cruce} y mutación p={prob_mutacion}/gen (módulo 7), "
            f"reemplazo de los {hijos_por_generacion} peores (módulo 8).")
        print(f"{'Gen':>4} {'Mejor':<14} {'Costo mejor':>12} {'Promedio':>10} {'Peor':>10}")
    for g in range(1, generaciones + 1):
        hijos = []
        while len(hijos) < hijos_por_generacion:
            p1 = seleccion_torneo(poblacion, terreno, rng, torneo)
            p2 = seleccion_torneo(poblacion, terreno, rng, torneo)
            if rng.random() < prob_cruce:
                h1, h2, _ = cruce_un_punto(p1, p2, rng)
            else:
                h1, h2 = p1, p2
            hijos.extend([mutacion(h1, rng, prob_mutacion), mutacion(h2, rng, prob_mutacion)])
        hijos = hijos[:hijos_por_generacion]
        evaluaciones += len(hijos)
        consultados.update(hijos)
        # Reemplazo: los hijos sustituyen a los peores individuos de la población.
        poblacion.sort(key=lambda e: (costo_de(e), e))
        poblacion = poblacion[:n - len(hijos)] + hijos
        costos = [costo_de(e) for e in poblacion]
        mejor_gen = min(poblacion, key=lambda e: (costo_de(e), e))
        if costo_de(mejor_gen) < costo_de(mejor) - EPS:
            mejor = mejor_gen
        promedio = statistics.mean(costos)
        traza.append({"generacion": g, "mejor": mejor_gen, "mejor_costo": costo_de(mejor_gen),
                      "promedio": promedio})
        if verbose:
            print(f"{g:4d} {vector(mejor_gen):<14} {costo_de(mejor_gen):12.6f} {promedio:10.6f} {max(costos):10.6f}")
    costo, acc = terreno.evaluar(mejor)
    if verbose:
        print(f"MEJOR GLOBAL TRAS {generaciones} GENERACIONES: {vector(mejor)} | costo={costo:.6f} | acc={acc:.2%}")
    return Resultado("Algoritmo Genético", traza[0]["mejor"], mejor, costo, acc,
                     generaciones, generaciones * hijos_por_generacion, evaluaciones,
                     len(consultados), traza, mejor, costo, 0, semilla)
