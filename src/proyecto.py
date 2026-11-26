"""Proyecto integrador de Inteligencia Artificial · Equipo 1.

Selección de modelos para un ensemble que detecta posibles fallas de maquinaria.
Módulos 1-4: escenario, Hill Climbing, Simulated Annealing y comparación.
Módulos 5-9: algoritmo genético y comparación global.
Módulo 10: CSP sobre las soluciones de HC, SA y AG.

Archivos: escenario.py (datos y costo), busqueda.py (HC y SA), genetico.py (AG),
csp.py (CSP) y modelos.py (los 6 clasificadores). Este archivo contiene el menú.
"""
import argparse
import json
import platform
import random
import statistics
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

from busqueda import hill_climbing, simulated_annealing
from csp import CSPEnsemble, ejecutar_modulo_10
from escenario import (ALFA, BETA, EPS, ESTADO_INICIAL, RAIZ, SEMILLA_DATOS, Terreno,
                       dividir_datos, exactitud, generar_datos, validar_estado, vecinos,
                       vector, votacion)
from genetico import (N_AG, PROB_CRUCE_AG, PROB_MUTACION_AG, SEMILLA_AG, TORNEO_AG,
                      algoritmo_genetico, poblacion_inicial, reparar)
from modelos import Estandarizador, crear_modelos


class Proyecto:
    def __init__(self, inicial=ESTADO_INICIAL, semilla_sa=7):
        self.semilla_sa = semilla_sa
        self.terreno = None
        self.resultado_ag = None
        self.resultado_hc = self.resultado_sa = None
        self.costo_final_hc = self.costo_final_sa = None
        validar_estado(inicial, 6)
        self.inicial = tuple(inicial)

    # MÓDULO 1: INICIALIZACIÓN Y DATOS DEL ESCENARIO
    def modulo_1(self):
        print("\nMÓDULO 1 · INICIALIZACIÓN Y DATOS DEL ESCENARIO")
        print("Selección de un ensemble para detectar posibles fallas de maquinaria.")
        print("Datos sintéticos: temperatura, vibración, presión y corriente; clases 0/1.")
        print("Se entrenan seis clasificadores implementados en Python sobre estas muestras simuladas.")
        X, y = generar_datos()
        train, val, test = dividir_datos(y)
        self.indices = {"entrenamiento": train, "validacion": val, "prueba": test}
        escalador = Estandarizador().fit([X[i] for i in train])
        Xtrain = escalador.transform([X[i] for i in train])
        self.Xval = escalador.transform([X[i] for i in val])
        self.Xtest = escalador.transform([X[i] for i in test])
        ytrain = [y[i] for i in train]
        self.yval, self.ytest = [y[i] for i in val], [y[i] for i in test]
        self.modelos = crear_modelos()
        predicciones = []
        print(f"Muestras: {len(X)}; entrenamiento={len(train)}, validación={len(val)}, prueba={len(test)}.")
        for identificador, nombre, modelo in self.modelos:
            modelo.fit(Xtrain, ytrain)
            pred = modelo.predict(self.Xval)
            predicciones.append(pred)
            print(f"  {identificador} · {nombre:<29} | accuracy validación={exactitud(self.yval, pred):.2%}")
        self.terreno = Terreno(predicciones, self.yval)
        # Al recargar se invalidan los resultados de ejecuciones anteriores.
        self.resultado_hc = self.resultado_sa = self.resultado_ag = None
        self.costo_final_hc = self.costo_final_sa = None
        print(f"Costo(E) = {ALFA} × (1 - Accuracy_validación(E)) + {BETA} × |E|/6.")
        print("Menor costo = menor error y/o menos modelos. El tiempo se reporta por separado.")
        print(f"Estado inicial compartido: {vector(self.inicial)}. Espacio: 63 subconjuntos no vacíos.")
        print("El inicio predeterminado es un caso didáctico elegido para evidenciar un mínimo local.")
        print("Vecindad: activar/desactivar un modelo. Votación: mayoría; empate -> clase 0.")

    def listo(self):
        """Los módulos 2-10 necesitan el escenario; si falta, se crea primero."""
        if self.terreno is None:
            print("\n(El escenario aún no existe: se ejecuta primero el módulo 1.)")
            self.modulo_1()
        return True

    def asegurar_resultados(self, hc=True, sa=True, ag=True):
        """Si un módulo necesita resultados que aún no se calculan, los obtiene sin
        detalle y con la configuración predeterminada, en lugar de detenerse."""
        faltantes = []
        if hc and self.resultado_hc is None:
            self.resultado_hc = hill_climbing(self.terreno, self.inicial, verbose=False)
            self.costo_final_hc = self.resultado_hc.costo
            faltantes.append("Hill Climbing (módulo 2)")
        if sa and self.resultado_sa is None:
            self.resultado_sa = simulated_annealing(self.terreno, self.inicial,
                                                     semilla=self.semilla_sa, verbose=False)
            self.costo_final_sa = self.resultado_sa.costo
            faltantes.append("Simulated Annealing (módulo 3)")
        if ag and self.resultado_ag is None:
            self.resultado_ag = algoritmo_genetico(
                self.terreno, n=N_AG, prob_cruce=PROB_CRUCE_AG, prob_mutacion=PROB_MUTACION_AG,
                torneo=TORNEO_AG, semilla=SEMILLA_AG, verbose=False)
            faltantes.append("Algoritmo Genético (módulo 8)")
        if faltantes:
            print("\n(Se ejecutaron sin detalle, con su configuración predeterminada: "
                  + ", ".join(faltantes) + ")")

    def modulo_2(self):
        if self.listo():
            self.resultado_hc = hill_climbing(self.terreno, self.inicial)
            self.costo_final_hc = self.resultado_hc.costo

    def modulo_3(self, detallado=False):
        if self.listo():
            self.resultado_sa = simulated_annealing(self.terreno, self.inicial,
                                                     semilla=self.semilla_sa, detallado=detallado)
            self.costo_final_sa = self.resultado_sa.costo

    # Los módulos 5, 6 y 7 reproducen, con la MISMA semilla del AG, los primeros
    # pasos que el módulo 8 ejecuta dentro de su ciclo: lo que se muestra aquí es
    # exactamente lo que ocurre al inicio de la generación 1.
    def _inicio_ag(self):
        rng = random.Random(SEMILLA_AG)
        return rng, poblacion_inicial(self.terreno, N_AG, rng)

    # MÓDULO 5: INICIALIZACIÓN DE LA POBLACIÓN
    def modulo_5(self):
        if not self.listo():
            return
        _, poblacion = self._inicio_ag()
        costo_de = lambda e: self.terreno.evaluar(e)[0]
        print("\nMÓDULO 5 · INICIALIZACIÓN DE LA POBLACIÓN")
        print(f"N={N_AG} cromosomas aleatorios de 6 bits (gen i = modelo Mi activo); "
              f"semilla={SEMILLA_AG}.")
        print("Un cromosoma con todos los genes en 0 se repara activando un modelo al azar.")
        for i, ind in enumerate(poblacion, 1):
            c, acc = self.terreno.evaluar(ind)
            print(f"  Ind {i:2d}: {vector(ind)} | costo={c:.6f} | acc={acc:.2%}")
        mejor = min(poblacion, key=lambda e: (costo_de(e), e))
        promedio = statistics.mean(costo_de(e) for e in poblacion)
        print(f"MEJOR DE LA GENERACIÓN 0: {vector(mejor)} | costo={costo_de(mejor):.6f} "
              f"| promedio poblacional={promedio:.6f}")

    # MÓDULO 6: SELECCIÓN DE PADRES POR TORNEO
    def modulo_6(self, verbose=True):
        if not self.listo():
            return None
        rng, poblacion = self._inicio_ag()
        costo_de = lambda e: self.terreno.evaluar(e)[0]
        if verbose:
            print(f"\nMÓDULO 6 · SELECCIÓN DE PADRES (torneo k={TORNEO_AG})")
            print("Se toman 3 individuos al azar de la población (módulo 5) y gana el de menor costo.")
            print("Primer par de padres de la generación 1 (misma semilla que el módulo 8):")
        padres = []
        for numero in (1, 2):
            # Mismo procedimiento que seleccion_torneo(), mostrando a los competidores.
            competidores = rng.sample(poblacion, TORNEO_AG)
            ganador = min(competidores, key=lambda e: (costo_de(e), e))
            padres.append(ganador)
            if verbose:
                print(f"  Torneo del padre {numero}:")
                for e in competidores:
                    print(f"    {vector(e)} | costo={costo_de(e):.6f}"
                          + ("  <- GANA" if e == ganador else ""))
        if verbose:
            print(f"PADRES SELECCIONADOS: P1={vector(padres[0])}  P2={vector(padres[1])}")
        return rng, padres

    # MÓDULO 7: OPERADORES GENÉTICOS (CRUCE Y MUTACIÓN)
    def modulo_7(self):
        if not self.listo():
            return
        rng, (p1, p2) = self.modulo_6(verbose=False)
        print(f"\nMÓDULO 7 · OPERADORES GENÉTICOS (cruce de un punto p={PROB_CRUCE_AG}, "
              f"mutación p={PROB_MUTACION_AG} por gen)")
        print(f"Padres obtenidos en el módulo 6: P1={vector(p1)}  P2={vector(p2)}")
        # Mismo orden de números aleatorios que cruce_un_punto() y mutacion().
        u = rng.random()
        if u < PROB_CRUCE_AG:
            punto = rng.randint(1, len(p1) - 1)
            h1, h2 = p1[:punto] + p2[punto:], p2[:punto] + p1[punto:]
            print(f"Cruce: u={u:.4f} < {PROB_CRUCE_AG} -> SÍ hay cruce; corte después del gen {punto}.")
            print(f"  H1 = P1[genes 1-{punto}] + P2[genes {punto + 1}-6] = {vector(h1)}")
            print(f"  H2 = P2[genes 1-{punto}] + P1[genes {punto + 1}-6] = {vector(h2)}")
            h1, h2 = reparar(h1, rng), reparar(h2, rng)
        else:
            h1, h2 = p1, p2
            print(f"Cruce: u={u:.4f} >= {PROB_CRUCE_AG} -> NO hay cruce; los hijos copian a los padres.")
        for nombre, hijo in (("H1", h1), ("H2", h2)):
            print(f"Mutación de {nombre} {vector(hijo)} (muta si u < {PROB_MUTACION_AG}):")
            nuevo = []
            for gen, valor in enumerate(hijo, 1):
                u = rng.random()
                muta = u < PROB_MUTACION_AG
                nuevo.append(1 - valor if muta else valor)
                print(f"    gen {gen}: valor={valor}  u={u:.4f}  "
                      + (f"MUTA -> {1 - valor}" if muta else "se conserva"))
            final = reparar(nuevo, rng)
            if final != tuple(nuevo):
                print("    (quedó en ceros: se reparó activando un modelo al azar)")
            c, acc = self.terreno.evaluar(final)
            print(f"  {nombre} final: {vector(final)} | costo={c:.6f} | acc={acc:.2%}")

    # MÓDULO 8: CICLO GENERACIONAL Y REEMPLAZO (AG COMPLETO)
    def modulo_8(self):
        if self.listo():
            self.resultado_ag = algoritmo_genetico(
                self.terreno, n=N_AG, prob_cruce=PROB_CRUCE_AG, prob_mutacion=PROB_MUTACION_AG,
                torneo=TORNEO_AG, semilla=SEMILLA_AG)

    # MÓDULO 9: COMPARACIÓN GLOBAL DEFINITIVA
    def modulo_9(self):
        if not self.listo():
            return
        self.asegurar_resultados()
        optimo = min(self.terreno.evaluar(e)[0]
                     for e in product((0, 1), repeat=self.terreno.m) if any(e))
        print("\nMÓDULO 9 · COMPARACIÓN GLOBAL DEFINITIVA (mismo escenario)")
        print(f"{'Algoritmo':<20} {'Vector':<14} {'Costo':>9} {'Acc.val':>8} {'Acc.test':>8} "
              f"{'Modelos':>7} {'Iter/Gen':>8} {'Eval.':>6} {'¿Global?':>8}")
        for r in (self.resultado_hc, self.resultado_sa, self.resultado_ag):
            acc_test = exactitud(self.ytest, self.predecir(r.estado, self.Xtest))
            print(f"{r.algoritmo:<20} {vector(r.estado):<14} {r.costo:9.6f} {r.accuracy:8.2%} "
                  f"{acc_test:8.2%} {sum(r.estado):7d} {r.iteraciones:8d} {r.evaluaciones:6d} "
                  f"{'Sí' if abs(r.costo - optimo) <= EPS else 'No':>8}")
        print(f"Óptimo global verificado por búsqueda exhaustiva: costo={optimo:.6f}.")
        print("HC: óptimo local de referencia | SA: búsqueda estocástica con temperatura | "
              "AG: mejor individuo tras G generaciones.")

    # MÓDULO 10: CSP SOBRE LAS SOLUCIONES DE HC, SA Y AG
    def modulo_10(self):
        if not self.listo():
            return
        self.asegurar_resultados()
        accuracies = [exactitud(self.yval, p) for p in self.terreno.predicciones]
        csp = CSPEnsemble(self.modelos, accuracies)
        ejecutar_modulo_10(csp, self.terreno,
                           [self.resultado_hc, self.resultado_sa, self.resultado_ag])

    def predecir(self, estado, X):
        return votacion([m.predict(X) for (_, _, m), activo in zip(self.modelos, estado) if activo])

    def medir_inferencia(self, estado, repeticiones=7):
        self.predecir(estado, self.Xval)  # Calentamiento, no cronometrado.
        tiempos = []
        for _ in range(repeticiones):
            inicio = time.perf_counter()
            self.predecir(estado, self.Xval)  # Incluye modelos + votación, sin caché.
            tiempos.append(1000 * (time.perf_counter() - inicio))
        return statistics.median(tiempos)

    # MÓDULO 4: COMPARACIÓN DE LOS RESULTADOS GUARDADOS
    def modulo_4(self, repeticiones=20, archivo=None):
        if not self.listo():
            return None
        self.asegurar_resultados(ag=False)
        print("\nMÓDULO 4 · COMPARACIÓN DE RESULTADOS GUARDADOS")
        # Oráculo SOLO de verificación: no participa en HC ni SA.
        exhaustivos = [(self.terreno.evaluar(e)[0], e)
                       for e in product((0, 1), repeat=self.terreno.m) if any(e)]
        costo_optimo = min(c for c, _ in exhaustivos)
        optimos = [e for c, e in exhaustivos if abs(c - costo_optimo) <= EPS]
        hc, sa = self.resultado_hc, self.resultado_sa
        filas = []
        for nombre, estado, resultado in [
            ("Todos los modelos", (1,) * self.terreno.m, None),
            ("Hill Climbing", hc.estado, hc),
            ("Simulated Annealing", sa.estado, sa),
        ]:
            costo, acc = self.terreno.evaluar(estado)
            ms = self.medir_inferencia(estado)
            filas.append({"algoritmo": nombre, "estado": estado, "costo": costo,
                          "accuracy_validacion": acc,
                          "accuracy_prueba": exactitud(self.ytest, self.predecir(estado, self.Xtest)),
                          "numero_modelos": sum(estado), "inferencia_ms_lote": ms,
                          "inferencia_ms_muestra": ms / len(self.Xval),
                          "iteraciones": resultado.iteraciones if resultado else None,
                          "movimientos": resultado.movimientos if resultado else None,
                          "evaluaciones": resultado.evaluaciones if resultado else None,
                          "combinaciones_unicas": resultado.combinaciones_unicas if resultado else None})
        print(f"{'Algoritmo':<20} {'Vector':<14} {'Costo':>9} {'Acc.val':>8} {'Acc.test':>8} {'Modelos':>7} {'Iter.':>6}")
        for fila in filas:
            print(f"{fila['algoritmo']:<20} {vector(fila['estado']):<14} {fila['costo']:9.6f} "
                  f"{fila['accuracy_validacion']:8.2%} {fila['accuracy_prueba']:8.2%} "
                  f"{fila['numero_modelos']:7d} {str(fila['iteraciones'] or '-'):>6}")
        print(f"\n{'Algoritmo':<20} {'Movim.':>7} {'Eval.':>7} {'Únicas':>7} {'ms/lote':>11} {'ms/muestra':>12}")
        for fila in filas:
            print(f"{fila['algoritmo']:<20} {str(fila['movimientos'] if fila['movimientos'] is not None else '-'):>7} "
                  f"{str(fila['evaluaciones'] if fila['evaluaciones'] is not None else '-'):>7} "
                  f"{str(fila['combinaciones_unicas'] if fila['combinaciones_unicas'] is not None else '-'):>7} "
                  f"{fila['inferencia_ms_lote']:11.4f} "
                  f"{fila['inferencia_ms_muestra']:12.6f}")
        print(f"Inferencia: mediana de 7 lotes de {len(self.Xval)} muestras, modelos + votación.")
        print("Las evaluaciones incluyen el inicio y repeticiones; únicas cuenta vectores distintos.")
        print("HC cuenta rondas de vecindario; SA cuenta propuestas de un bit, incluso vacías rechazadas.")
        print(f"\nVerificación exhaustiva independiente: {len(exhaustivos)} combinaciones.")
        print(f"Óptimo global de ESTE terreno de validación: costo={costo_optimo:.6f}.")
        print("Vector(es) óptimo(s): " + "; ".join(map(vector, optimos)))
        es_local = all(self.terreno.evaluar(e)[0] >= hc.costo - EPS for e in vecinos(hc.estado))
        print(f"HC sin vecinos mejores: {es_local}; brecha al global={hc.costo - costo_optimo:.6f}.")
        alcanzado = abs(sa.costo - costo_optimo) <= EPS
        print(f"SA alcanzó un óptimo global: {alcanzado}; mejora de costo frente a HC={hc.costo-sa.costo:.6f}.")
        primera_llegada = next((p["iteracion"] for p in sa.traza
                                if abs(p["mejor_costo"] - costo_optimo) <= EPS), None)
        if abs(self.terreno.evaluar(sa.inicial)[0] - costo_optimo) <= EPS:
            primera_llegada = 0
        if primera_llegada is not None:
            print(f"SA encontró el global por primera vez en el paso {primera_llegada}; "
                  f"completó {sa.iteraciones} propuestas de enfriamiento.")
        print("SA no garantiza alcanzar el óptimo global en toda ejecución finita.")
        escapes = []
        for paso in sa.traza:
            if (paso["aceptado"] and paso["delta_f"] < -EPS
                    and all(self.terreno.evaluar(v)[0] >= paso["costo_desde"] - EPS
                            for v in vecinos(paso["desde"]))):
                escapes.append(paso)
        if escapes:
            p = escapes[0]
            print(f"Escape verificado en i={p['iteracion']}: {vector(p['desde'])} "
                  f"(mínimo local, {p['costo_desde']:.6f}) -> {vector(p['candidato'])} "
                  f"({p['costo_candidato']:.6f}); p={p['probabilidad']:.6f}, u={p['aleatorio']:.6f}.")
        escape_hc = next((p for p in escapes if p["desde"] == hc.estado), None)
        if escape_hc:
            p = escape_hc
            print(f"SA también salió del mismo mínimo donde se detuvo HC, en i={p['iteracion']}: "
                  f"{vector(p['desde'])} ({p['costo_desde']:.6f}) -> "
                  f"{vector(p['candidato'])} ({p['costo_candidato']:.6f}); "
                  f"p={p['probabilidad']:.6f} > u={p['aleatorio']:.6f}.")
        consistencia = self.consistencia(costo_optimo, repeticiones)
        informe = {"escenario": "Detección simulada de fallas de maquinaria",
                   "datos": "Sintéticos; 600 observaciones, 4 atributos, 2 clases",
                   "python": platform.python_version(), "plataforma": platform.platform(),
                   "semilla_datos": SEMILLA_DATOS, "semilla_particion": 17,
                   "particiones": {k: len(v) for k, v in self.indices.items()},
                   "alfa": ALFA, "beta": BETA, "inicial": self.inicial,
                   "sa_config": {"T0": 0.08, "enfriamiento": 0.99,
                                 "Tmin": 0.0001, "max_iteraciones": 1000},
                   "modelos": [{"id": i, "nombre": n} for i, n, _ in self.modelos],
                   "etiquetas_validacion": self.yval,
                   "predicciones_validacion_por_modelo": self.terreno.predicciones,
                   "tabla": filas, "hill_climbing": asdict(hc),
                   "simulated_annealing": asdict(sa), "costo_optimo_global": costo_optimo,
                   "estados_optimos": optimos, "evaluaciones_verificacion": len(exhaustivos),
                   "sa_primera_llegada_global": primera_llegada,
                   "escapes_verificados": escapes, "consistencia": consistencia}
        if archivo:
            destino = Path(archivo)
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nResultados y trazas completas guardados en: {destino}")
        return informe

    def consistencia(self, costo_optimo, repeticiones):
        if repeticiones < 1:
            raise ValueError("La cantidad de repeticiones debe ser positiva.")
        ejecuciones = []
        for semilla in range(repeticiones):
            hc = hill_climbing(self.terreno, self.inicial, verbose=False)
            sa = simulated_annealing(self.terreno, self.inicial, semilla=semilla, verbose=False)
            ejecuciones.append({"semilla_sa": semilla, "hc_costo": hc.costo,
                                "sa_costo": sa.costo, "hc_estado": hc.estado, "sa_estado": sa.estado})
        resumen = {}
        print(f"\nConsistencia: {repeticiones} ejecuciones, mismos datos/modelos/inicio; semillas SA 0..{repeticiones-1}.")
        for etiqueta, clave in [("Hill Climbing", "hc_costo"), ("Simulated Annealing", "sa_costo")]:
            costos = [e[clave] for e in ejecuciones]
            globales = sum(abs(c - costo_optimo) <= EPS for c in costos)
            resumen[etiqueta] = {"costo_medio": statistics.mean(costos),
                                "desviacion_poblacional": statistics.pstdev(costos),
                                "minimo": min(costos), "maximo": max(costos),
                                "veces_optimo_global": globales}
            print(f"  {etiqueta}: promedio={statistics.mean(costos):.6f}; "
                f"desv.={statistics.pstdev(costos):.6f}; global={globales}/{repeticiones}.")
        print("Esto mide variación de búsqueda; no variación por nuevos datos o reentrenamiento.")
        return {"resumen": resumen, "ejecuciones": ejecuciones}


MENU = [
    ("1", "Módulo 1 · Inicializar escenario y entrenar los 6 modelos"),
    ("2", "Módulo 2 · Hill Climbing"),
    ("3", "Módulo 3 · Simulated Annealing"),
    ("4", "Módulo 4 · Comparar resultados (HC vs SA)"),
    ("5", "Módulo 5 · AG: inicialización de la población"),
    ("6", "Módulo 6 · AG: selección de padres por torneo"),
    ("7", "Módulo 7 · AG: cruce y mutación"),
    ("8", "Módulo 8 · AG: ciclo generacional completo y reemplazo"),
    ("9", "Módulo 9 · Comparación global definitiva (HC, SA, AG)"),
    ("10", "Módulo 10 · CSP: restricciones y factibilidad de HC, SA y AG"),
    ("0", "Salir"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true",
                        help="Ejecutar todos los módulos en orden, sin menú.")
    parser.add_argument("--detallado", action="store_true", help="Imprimir todas las propuestas de SA.")
    parser.add_argument("--semilla-sa", type=int, default=7)
    parser.add_argument("--inicio", default="".join(map(str, ESTADO_INICIAL)),
                        help="Seis bits no vacíos, en orden M1..M6; predeterminado: 110011.")
    parser.add_argument("--repeticiones", type=int, default=20)
    parser.add_argument("--salida", type=Path, default=RAIZ / "resultados" / "resultados.json",
                        help="Archivo JSON con las trazas del módulo 4.")
    args = parser.parse_args()
    if args.repeticiones < 1:
        parser.error("--repeticiones debe ser mayor que cero.")
    if len(args.inicio) != 6 or any(bit not in "01" for bit in args.inicio) or "1" not in args.inicio:
        parser.error("--inicio debe contener seis bits y al menos un 1 (ejemplo: 110011).")
    proyecto = Proyecto(tuple(map(int, args.inicio)), semilla_sa=args.semilla_sa)
    acciones = {
        "1": proyecto.modulo_1,
        "2": proyecto.modulo_2,
        "3": lambda: proyecto.modulo_3(args.detallado),
        "4": lambda: proyecto.modulo_4(args.repeticiones, args.salida),
        "5": proyecto.modulo_5,
        "6": proyecto.modulo_6,
        "7": proyecto.modulo_7,
        "8": proyecto.modulo_8,
        "9": proyecto.modulo_9,
        "10": proyecto.modulo_10,
    }
    if args.demo:
        for opcion in acciones:
            acciones[opcion]()
        return
    while True:
        print("\nPROYECTO INTEGRADOR · SELECCIÓN DE MODELOS PARA UN ENSEMBLE")
        for opcion, texto in MENU:
            print(f"{opcion:>3}. {texto}")
        try:
            opcion = input("Selecciona una opción: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin del programa.")
            break
        if opcion == "0":
            print("Fin del programa.")
            break
        if opcion in acciones:
            acciones[opcion]()
        else:
            print("Opción inválida. Escribe un número del 0 al 10.")


if __name__ == "__main__":
    main()
