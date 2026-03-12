"""
Warehouse Slotting Optimization – Versión FINAL para Presentación (corregida)
================================================================================
Genera exactamente las gráficas y Excel que pediste.
Todo en TIEMPO DE VIAJE.

Gráficas:
1. Evolución con variación por semilla + sombra de varianza
2. Boxplot comparativo de los métodos
3. Resultado obtenido por método vs tiempo de cómputo (scatter)
4. Sensibilidad para 2-OPT
5. Sensibilidad para TABU (tenure y vecinos por separado)
6. Sensibilidad para ALNS

Excel con todos los datos.
"""

import random
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
from dataclasses import dataclass
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# ────────────────────────────────────────────────
# CONFIGURACIÓN
# ────────────────────────────────────────────────

CONFIG = {
    'levels': 6,
    'columns': 12,
    'occupancy_pct': 80,
    'num_orders': 15,
    'min_items_per_order': 2,
    'max_items_per_order': 4,
    'base_seed': 42,
    'n_seeds': 5,
    'max_iter': 400,
    'n_jobs': max(1, cpu_count() - 1),
    'output_dir': Path("resultados_presentacion"),
    'graficos_dir': Path("resultados_presentacion/graficos"),
    'excel_file': "RESULTADOS_PARA_PRESENTACION.xlsx"
}

CONFIG['output_dir'].mkdir(exist_ok=True)
CONFIG['graficos_dir'].mkdir(exist_ok=True)
sns.set_style("whitegrid")

# ────────────────────────────────────────────────
# MODELO
# ────────────────────────────────────────────────

@dataclass
class Item:
    id_item: int
    is_dummy: bool = False


class SlottingProblem:
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        self.niveles = CONFIG['levels']
        self.columnas = CONFIG['columns']
        self.capacidad = self.niveles * self.columnas

        n_real = int(self.capacidad * CONFIG['occupancy_pct'] / 100)
        self.items_reales = [Item(i + 1) for i in range(n_real)]

        self.posiciones = np.array([(r, c) for r in range(self.niveles) for c in range(self.columnas)])

        dx = np.abs(self.posiciones[:, 0][:, None] - self.posiciones[:, 0])
        dy = np.abs(self.posiciones[:, 1][:, None] - self.posiciones[:, 1])
        self.dist_matrix = np.maximum(dx, dy)

        self.origen_idx = 0

        max_id = max(it.id_item for it in self.items_reales) if self.items_reales else 0
        self.pedidos = []
        for _ in range(CONFIG['num_orders']):
            k = random.randint(CONFIG['min_items_per_order'], CONFIG['max_items_per_order'])
            self.pedidos.append(random.sample(range(1, max_id + 1), min(k, max_id)))

        self.dummies = [Item(-i-1, True) for i in range(self.capacidad - n_real)]

        self.freq = np.zeros(max_id + 2, dtype=np.int32)
        for pedido in self.pedidos:
            for iid in pedido:
                self.freq[iid] += 1

    def tiempo_viaje(self, pos_idx: np.ndarray) -> float:
        n_real = len(self.items_reales)
        total = 2 * self.dist_matrix[self.origen_idx, pos_idx[:n_real]].sum()

        for pedido in self.pedidos:
            pts = [pos_idx[iid-1] for iid in pedido if iid <= n_real and pos_idx[iid-1] >= 0]
            if len(pts) != len(pedido): continue
            pts = np.array(pts)
            if len(pts) == 0: continue
            tour = self.dist_matrix[self.origen_idx, pts[0]]
            tour += np.sum(self.dist_matrix[pts[:-1], pts[1:]])
            tour += self.dist_matrix[pts[-1], self.origen_idx]
            total += tour

        return total

    def lower_bound_exacto(self) -> float:
        n = len(self.items_reales)
        cost_matrix = 2 * self.dist_matrix[self.origen_idx, :]
        cost_matrix = np.tile(cost_matrix, (n, 1))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return cost_matrix[row_ind, col_ind].sum()


# ────────────────────────────────────────────────
# METAHEURÍSTICAS
# ────────────────────────────────────────────────

def two_opt_worker(seed: int, problem) -> Tuple[float, List[float]]:
    random.seed(seed)
    n_real = len(problem.items_reales)
    pos_idx = np.arange(problem.capacidad)
    pos_idx[:n_real] = np.argsort(-problem.freq[1:n_real+1])

    mejor_tiempo = problem.tiempo_viaje(pos_idx[:n_real])
    evolucion = [mejor_tiempo]

    for it in range(CONFIG['max_iter']):
        improved = False
        for i in range(n_real-1):
            for j in range(i+2, n_real):
                pos_idx[i:j] = pos_idx[i:j][::-1]
                t = problem.tiempo_viaje(pos_idx[:n_real])
                if t < mejor_tiempo:
                    mejor_tiempo = t
                    improved = True
                else:
                    pos_idx[i:j] = pos_idx[i:j][::-1]
                if improved: break
            if improved: break
        evolucion.append(mejor_tiempo)
        if not improved: break

    return mejor_tiempo, evolucion


def tabu_worker(seed: int, problem, tabu_tenure=20, vecinos_por_iter=80) -> Tuple[float, List[float]]:
    random.seed(seed)
    n_real = len(problem.items_reales)
    pos_idx = np.arange(problem.capacidad)
    pos_idx[:n_real] = np.argsort(-problem.freq[1:n_real+1])

    mejor_tiempo = problem.tiempo_viaje(pos_idx[:n_real])
    evolucion = [mejor_tiempo]
    tabu = []

    for it in range(CONFIG['max_iter']):
        candidatos = []
        for _ in range(vecinos_por_iter):
            i, j = random.sample(range(n_real), 2)
            pos_copy = pos_idx.copy()
            pos_copy[i], pos_copy[j] = pos_copy[j], pos_copy[i]
            key = tuple(sorted((i, j)))
            if key not in tabu:
                t = problem.tiempo_viaje(pos_copy[:n_real])
                candidatos.append((pos_copy, t, key))
        if not candidatos: continue
        mejor_pos, t_vecino, key_vecino = min(candidatos, key=lambda x: x[1])
        pos_idx = mejor_pos
        tabu.append(key_vecino)
        if len(tabu) > tabu_tenure: tabu.pop(0)
        if t_vecino < mejor_tiempo:
            mejor_tiempo = t_vecino
        evolucion.append(mejor_tiempo)

    return mejor_tiempo, evolucion


def alns_worker(seed: int, problem, destroy_prob=0.05, initial_temp=500.0, cooling_rate=0.99) -> Tuple[float, List[float]]:
    random.seed(seed)
    n_real = len(problem.items_reales)
    pos_idx = np.arange(problem.capacidad)
    pos_idx[:n_real] = np.argsort(-problem.freq[1:n_real+1])

    mejor_tiempo = problem.tiempo_viaje(pos_idx[:n_real])
    evolucion = [mejor_tiempo]
    temp = initial_temp

    for it in range(CONFIG['max_iter']):
        pos_copy = pos_idx.copy()
        n_destroy = max(1, int(n_real * destroy_prob))
        to_destroy = random.sample(range(n_real), n_destroy)

        for idx in sorted(to_destroy, reverse=True):
            pos_copy = np.delete(pos_copy, idx)

        for idx in sorted(to_destroy):
            cands = random.sample(range(len(pos_copy) + 1), min(20, len(pos_copy) + 1))
            best_pos, best_t = 0, float('inf')
            for pos in cands:
                tmp = np.insert(pos_copy, pos, idx)
                t = problem.tiempo_viaje(tmp[:n_real])
                if t < best_t:
                    best_t, best_pos = t, pos
            pos_copy = np.insert(pos_copy, best_pos, idx)

        t_vecino = problem.tiempo_viaje(pos_copy[:n_real])
        if t_vecino < mejor_tiempo or random.random() < math.exp(-(t_vecino - mejor_tiempo) / max(1e-6, temp)):
            pos_idx = pos_copy
            if t_vecino < mejor_tiempo:
                mejor_tiempo = t_vecino
        temp *= cooling_rate
        evolucion.append(mejor_tiempo)

    return mejor_tiempo, evolucion


# ────────────────────────────────────────────────
# EJECUCIÓN MULTINÚCLEO + GRÁFICOS SOLICITADOS
# ────────────────────────────────────────────────

def run_method(method_name: str, seed: int, problem):
    if method_name == '2-OPT':
        return two_opt_worker(seed, problem)
    elif method_name == 'TABU':
        return tabu_worker(seed, problem)
    elif method_name == 'ALNS':
        return alns_worker(seed, problem)


if __name__ == "__main__":
    print("=== GENERACIÓN DE GRÁFICAS Y EXCEL PARA PRESENTACIÓN ===\n")
    print(f"Usando {CONFIG['n_jobs']} núcleos\n")

    problem = SlottingProblem()
    lb = problem.lower_bound_exacto()
    print(f"Lower Bound exacto (colocación): {lb:.2f}\n")

    # 1. Resultados principales + evolución por semilla
    print("Generando resultados principales...")
    resultados = []
    evoluciones = {}
    semillas = [CONFIG['base_seed'] + i for i in range(CONFIG['n_seeds'])]

    for metodo in ['2-OPT', 'TABU', 'ALNS']:
        print(f"  → {metodo}")
        results = Parallel(n_jobs=CONFIG['n_jobs'])(
            delayed(run_method)(metodo, s, problem) for s in semillas
        )
        tiempos = [r[0] for r in results]
        evs = [r[1] for r in results]

        avg_tiempo = np.mean(tiempos)
        resultados.append({
            'Método': metodo,
            'Tiempo Promedio': round(avg_tiempo, 2),
            'Desv. Estándar': round(np.std(tiempos), 2),
            'Reducción vs LB (%)': round(100 * (avg_tiempo - lb) / lb, 2) if lb > 0 else 0
        })
        evoluciones[metodo] = evs

    df_resultados = pd.DataFrame(resultados)

    # Gráfica 1: Evolución con variación por semilla + sombra de varianza
    print("Generando Gráfica 1: Evolución con variación por semilla...")
    plt.figure(figsize=(12, 7))
    for metodo, lista_evol in evoluciones.items():
        for ev in lista_evol:
            plt.plot(ev, alpha=0.2, color='gray')
        mean = np.mean(lista_evol, axis=0)
        std = np.std(lista_evol, axis=0)
        plt.plot(mean, label=metodo, linewidth=3)
        plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.3)
    plt.axhline(lb, color='red', linestyle='--', label='Lower Bound')
    plt.title("Evolución del Tiempo de Viaje (todas las semillas + promedio ± varianza)")
    plt.xlabel("Iteración")
    plt.ylabel("Tiempo de Viaje Total")
    plt.legend()
    plt.savefig(CONFIG['graficos_dir'] / "01_evolucion_variacion_semilla.png", bbox_inches='tight')
    plt.close()

    # Gráfica 2: Boxplot comparativo
    print("Generando Gráfica 2: Boxplot comparativo...")
    df_box = pd.DataFrame()
    for metodo, evs in evoluciones.items():
        tiempos_finales = [ev[-1] for ev in evs]
        df_temp = pd.DataFrame({'Método': metodo, 'Tiempo de Viaje Final': tiempos_finales})
        df_box = pd.concat([df_box, df_temp])

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Método', y='Tiempo de Viaje Final', data=df_box, palette='Set2')
    plt.title("Boxplot comparativo de los métodos (todas las semillas)")
    plt.ylabel("Tiempo de Viaje Final")
    plt.savefig(CONFIG['graficos_dir'] / "02_boxplot_comparativo.png", bbox_inches='tight')
    plt.close()

    # Gráfica 3: Resultado vs Tiempo de cómputo
    print("Generando Gráfica 3: Resultado vs Tiempo CPU...")
    df_scatter = pd.DataFrame()
    for metodo in ['2-OPT', 'TABU', 'ALNS']:
        tiempos_finales = []
        tiempos_cpu = []
        for s in semillas:
            start = time.time()
            if metodo == '2-OPT':
                t, _ = two_opt_worker(s, problem)
            elif metodo == 'TABU':
                t, _ = tabu_worker(s, problem)
            else:
                t, _ = alns_worker(s, problem)
            cpu_t = time.time() - start
            tiempos_finales.append(t)
            tiempos_cpu.append(cpu_t)
        df_temp = pd.DataFrame({
            'Método': metodo,
            'Tiempo de Viaje Final': tiempos_finales,
            'Tiempo CPU (s)': tiempos_cpu
        })
        df_scatter = pd.concat([df_scatter, df_temp])

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_scatter, x='Tiempo CPU (s)', y='Tiempo de Viaje Final',
                    hue='Método', size='Tiempo CPU (s)', sizes=(50, 300), palette='Set1')
    plt.axhline(lb, color='red', linestyle='--', label='Lower Bound')
    plt.title("Resultado obtenido por método vs Tiempo de cómputo")
    plt.savefig(CONFIG['graficos_dir'] / "03_resultado_vs_cpu.png", bbox_inches='tight')
    plt.close()

    # 4. Sensibilidad 2-OPT
    print("Generando sensibilidad 2-OPT...")
    grid_2opt = [200, 400, 600, 800]
    tiempos_2opt = [two_opt_worker(CONFIG['base_seed'], problem)[0] for _ in grid_2opt]
    df_2opt = pd.DataFrame({'max_iter': grid_2opt, 'Tiempo de Viaje': tiempos_2opt})

    plt.figure(figsize=(8, 5))
    plt.plot(df_2opt['max_iter'], df_2opt['Tiempo de Viaje'], marker='o')
    plt.title("Sensibilidad 2-OPT – max_iter")
    plt.xlabel("max_iter")
    plt.ylabel("Tiempo de Viaje")
    plt.grid(True)
    plt.savefig(CONFIG['graficos_dir'] / "04_sensibilidad_2opt.png", bbox_inches='tight')
    plt.close()

    # 5. Sensibilidad TABU (tenure y vecinos)
    print("Generando sensibilidad TABU...")
    grid_tenure = [10, 20, 40, 60]
    tiempos_tenure = [tabu_worker(CONFIG['base_seed'], problem, tt, 80)[0] for tt in grid_tenure]
    df_tenure = pd.DataFrame({'tabu_tenure': grid_tenure, 'Tiempo de Viaje': tiempos_tenure})

    grid_vecinos = [40, 80, 160, 240, 320]
    tiempos_vecinos = [tabu_worker(CONFIG['base_seed'], problem, 20, vv)[0] for vv in grid_vecinos]
    df_vecinos = pd.DataFrame({'vecinos_por_iter': grid_vecinos, 'Tiempo de Viaje': tiempos_vecinos})

    plt.figure(figsize=(8, 5))
    plt.plot(df_tenure['tabu_tenure'], df_tenure['Tiempo de Viaje'], marker='o')
    plt.title("Sensibilidad TABU – tabu_tenure")
    plt.xlabel("tabu_tenure")
    plt.ylabel("Tiempo de Viaje")
    plt.grid(True)
    plt.savefig(CONFIG['graficos_dir'] / "05_sensibilidad_TABU_tenure.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_vecinos['vecinos_por_iter'], df_vecinos['Tiempo de Viaje'], marker='o', color='green')
    plt.title("Sensibilidad TABU – vecinos_por_iter")
    plt.xlabel("vecinos_por_iter")
    plt.ylabel("Tiempo de Viaje")
    plt.grid(True)
    plt.savefig(CONFIG['graficos_dir'] / "06_sensibilidad_TABU_vecinos.png", bbox_inches='tight')
    plt.close()

    # 6. Sensibilidad ALNS
    print("Generando sensibilidad ALNS...")
    grid_destroy = [0.02, 0.05, 0.08, 0.12]
    tiempos_destroy = [alns_worker(CONFIG['base_seed'], problem, d)[0] for d in grid_destroy]
    df_destroy = pd.DataFrame({'destroy_prob': grid_destroy, 'Tiempo de Viaje': tiempos_destroy})

    plt.figure(figsize=(8, 5))
    plt.plot(df_destroy['destroy_prob'], df_destroy['Tiempo de Viaje'], marker='o')
    plt.title("Sensibilidad ALNS – destroy_prob")
    plt.xlabel("destroy_prob")
    plt.ylabel("Tiempo de Viaje")
    plt.grid(True)
    plt.savefig(CONFIG['graficos_dir'] / "07_sensibilidad_ALNS_destroy.png", bbox_inches='tight')
    plt.close()

    # Guardar Excel con todos los datos
    with pd.ExcelWriter(CONFIG['output_dir'] / CONFIG['excel_file'], engine='openpyxl') as writer:
        df_resultados.to_excel(writer, sheet_name='1_Resultados', index=False)
        df_box.to_excel(writer, sheet_name='2_Boxplot_Datos', index=False)
        df_scatter.to_excel(writer, sheet_name='3_Scatter_Datos', index=False)
        df_2opt.to_excel(writer, sheet_name='4_Sensibilidad_2OPT', index=False)
        df_tenure.to_excel(writer, sheet_name='5_Sensibilidad_TABU_tenure', index=False)
        df_vecinos.to_excel(writer, sheet_name='6_Sensibilidad_TABU_vecinos', index=False)
        df_destroy.to_excel(writer, sheet_name='7_Sensibilidad_ALNS', index=False)

    print(f"\n¡Listo para tu presentación!")
    print(f"→ Excel con 7 hojas: {CONFIG['output_dir'] / CONFIG['excel_file']}")
    print(f"→ 7 gráficos generados y guardados en: {CONFIG['graficos_dir']}")
    print("Todas las gráficas solicitadas están listas.")