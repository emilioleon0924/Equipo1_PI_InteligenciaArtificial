"""Módulo 11: clasificación de máquinas a partir de su temperatura."""
import csv
from decimal import Decimal

from escenario import DATA_DIR


def ejecutar_modulo_11():
    temperaturas = {"normal": [], "posible_falla": []}
    with (DATA_DIR / "discriminante_entrenamiento.csv").open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            temperaturas[fila["clase"]].append(Decimal(fila["temperatura"]))

    print("\nMÓDULO 11 · DISCRIMINANTE LINEAL POR TEMPERATURA")
    medias = {}
    for clase, valores in temperaturas.items():
        total = sum(valores)
        medias[clase] = total / len(valores)
        print(f"Media de {clase}: {total} / {len(valores)} = {medias[clase]}")

    baja, alta = sorted(medias, key=medias.get)
    corte = (medias[baja] + medias[alta]) / 2
    print(f"Punto de corte: ({medias[baja]} + {medias[alta]}) / 2 = {corte}")
    print(f"Temperatura < {corte}: {baja}; en otro caso: {alta}.")
    with (DATA_DIR / "discriminante_casos_nuevos.csv").open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            temperatura = Decimal(fila["temperatura"])
            prediccion = baja if temperatura < corte else alta
            print(f"  {fila['id']} | temperatura={temperatura} | clase asignada: {prediccion}")
