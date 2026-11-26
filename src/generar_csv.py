"""Genera los archivos de data/ a partir del mismo escenario sintético del proyecto.

Los CSV ya vienen incluidos en el repositorio; este script solo documenta de
dónde salen y permite regenerarlos de forma reproducible:

    python3 src/generar_csv.py
"""
import csv

from escenario import DATA_DIR, dividir_datos, generar_datos

CLASES = {0: "normal", 1: "posible_falla"}


def escribir(nombre, encabezado, filas):
    destino = DATA_DIR / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)
    print(f"  {destino.relative_to(DATA_DIR.parent)}: {len(filas)} filas")


def main():
    X, y = generar_datos()
    entrenamiento, _, prueba = dividir_datos(y)
    print("Generando archivos de data/ (semilla de datos 2026, partición 17):")

    # Módulo 11: 10 máquinas por clase de la partición de entrenamiento, con
    # muestreo sistemático por deciles: se ordenan las temperaturas de la clase
    # y se toma el punto central de cada décima parte (muestra representativa).
    filas = []
    for clase in (0, 1):
        temperaturas = sorted(X[i][0] for i in entrenamiento if y[i] == clase)
        n = len(temperaturas)
        for k in range(10):
            temperatura = temperaturas[int((k + 0.5) * n / 10)]
            filas.append([f"E{len(filas) + 1:02d}", round(temperatura, 1), CLASES[clase]])
    escribir("discriminante_entrenamiento.csv", ["id", "temperatura", "clase"], filas)

    # Módulo 11: casos nuevos (3 por clase) de la partición de prueba.
    # La clase real solo se usa para comparar después de clasificar.
    filas, cuenta = [], {0: 0, 1: 0}
    for i in prueba:
        if cuenta[y[i]] < 3:
            cuenta[y[i]] += 1
            filas.append([f"N{len(filas) + 1:02d}", round(X[i][0], 1), CLASES[y[i]]])
    escribir("discriminante_casos_nuevos.csv", ["id", "temperatura", "clase_real"], filas)

    # Módulo 12: 24 máquinas de la partición de prueba (temperatura y vibración).
    # La clase real NO participa en K-medias; solo sirve para interpretar los grupos.
    filas = [[f"Q{n:02d}", round(X[i][0], 1), round(X[i][1], 2), CLASES[y[i]]]
             for n, i in enumerate(prueba[30:54], 1)]
    escribir("kmedias_maquinas.csv", ["id", "temperatura", "vibracion", "clase_real"], filas)


if __name__ == "__main__":
    main()
