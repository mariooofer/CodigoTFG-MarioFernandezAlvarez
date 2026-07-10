'''
VISUALIZACIÓN DE ÁREAS RELATIVAS POR TIEMPO


En el modo 0 hay ajuste exponencial a los cortes de la distribución de áreas,
en el modo 1 no hay ese ajuste.

INSTRUCCIONES DE USO:
    python verarea.py <directorio_simulacion> <modo>

'''


from sys import argv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

# Tamaños de fuente de ejes y números (afecta a todas las figuras)
plt.rcParams.update({
    "axes.labelsize": 15,   # etiquetas de los ejes (X, Y)
    "axes.titlesize": 16,   # títulos
    "xtick.labelsize": 13,  # números del eje X
    "ytick.labelsize": 13,  # números del eje Y
    "legend.fontsize": 13,  # leyenda
})
from scipy.optimize import curve_fit
from scipy.stats import kstest



#  CONTROL DE ENTRADA


if len(argv) in (3, 4):
    simdir = Path(argv[1])
    modo   = int(argv[2])
    ruta   = simdir / "areas_por_tiempov5.txt"
    if modo not in (0, 1):
        print("Error: el modo debe ser 0 o 1.")
        exit()
else:
    print("Uso: python verarea.py <directorio_simulacion> <modo>")
    print("     modo: 0 = estática, 1 = dinámica")
    exit()

if not ruta.exists():
    print(f"Error: No se encontró {ruta}")
    exit()

# Subcarpeta para las figuras (se crea si no existe)
figdir = simdir / "figuras_areasv7"
figdir.mkdir(exist_ok=True)

# Conversión del eje temporal según el modo
if modo == 0:
    DT = 0.0094           # s/frame
    ETIQUETA_TIEMPO = "Tiempo (s)"
else:
    DT = 8.25 / 60.0      # min/frame (8,25 s → minutos)
    ETIQUETA_TIEMPO = "Tiempo (min)"



#  LECTURA

with open(ruta, "r") as f:
    areas_por_tiempo = [
        [float(v) for v in linea.strip().split()]
        for linea in f
        if linea.strip()
    ]

areas_por_tiempo = areas_por_tiempo[1:]
n_frames = len(areas_por_tiempo)


#  PREPARAR ARRAYS Y CALCULAR INICIO_X

todas_areas  = []
todos_frames = []

for t, areas_frame in enumerate(areas_por_tiempo):
    for area in areas_frame:
        todas_areas.append(area)
        todos_frames.append(t)

todas_areas  = np.array(todas_areas)
todos_frames = np.array(todos_frames)

_areas_tmp = todas_areas[todas_areas > 0]
_counts_prev, _bin_edges_prev = np.histogram(_areas_tmp, bins=100, range=(0, 0.4))
_umbral = _counts_prev.max() * 0.05
# Ignorar el primer bin (índice 0) para evitar la columna de valores ~0
_bins_con_datos = np.where(_counts_prev > _umbral)[0]
_bins_con_datos = _bins_con_datos[_bins_con_datos > 0]  # excluir bin 0
inicio_x = _bin_edges_prev[_bins_con_datos[0]] if len(_bins_con_datos) > 0 else 0.01
print(f"Inicio del histograma: {inicio_x:.4f}")

# Número de bins en [inicio_x, 0.4] con la MISMA anchura que el histograma
# principal (50 bins en [inicio_x, 1]). Se usa en corte, calidad e insets.
_ancho_bin_ppal = (1 - inicio_x) / 50
N_BINS_04 = max(int(round((0.4 - inicio_x) / _ancho_bin_ppal)), 1)


#  HISTOGRAMA 2D 


fig, ax = plt.subplots(figsize=(10, 6))
tiempos_conv = todos_frames * DT
counts_2d, x_edges, y_edges, im = ax.hist2d(
    todas_areas, tiempos_conv,
    bins=[50, n_frames],
    range=[[inicio_x, 1], [0, n_frames * DT]],
    cmap="inferno",
)
ax.set_xlim(inicio_x, 0.3)

if modo == 1:
    ax.axhline(15 * DT, color="white", linestyle="--", linewidth=1.5,
               label="Inicio deformación", zorder=1)
    ax.axhline((15 + 256) * DT, color="white", linestyle="--", linewidth=1.5,
               label="Fin deformación", zorder=1)
    ax.legend(fontsize=9, loc="upper left")

    nombre_fig = "histograma_areas_dinamicov7.png"
else:
    nombre_fig = "histograma_areas100v7.png"


cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Número de clústeres", fontsize=15)
ax.set_xlabel("Área relativa")
ax.set_ylabel(ETIQUETA_TIEMPO)

# ── Insets: dos cortes de frames en la zona derecha (poca señal) ──
FRAMES_INSET = (30, 100, 200)   # tres frames mostrados como cortes en el histograma

def _fase_deformacion(frame_idx):
    """Devuelve la fase de deformación del frame (modo 1)."""
    if frame_idx < 15:
        return "antes de la deformación"
    elif frame_idx <= 273:
        return "durante la deformación"
    else:
        return "después de la deformación"

def _dibujar_inset_hist(bounds, frame_idx, show_xlabel=True, show_ylabel=True, fs=0):
    """Inset con el corte de un frame en escala log (barras + ajuste en modo 0).
    fs: incremento de tamaño de fuente respecto a la base (0 = base)."""
    if frame_idx >= len(areas_por_tiempo):
        return
    areas_i = np.array(areas_por_tiempo[frame_idx])
    areas_i = areas_i[areas_i >= inicio_x]
    if len(areas_i) < 2:
        return
    dens_i, edges_i = np.histogram(areas_i, bins=N_BINS_04,
                                   range=(inicio_x, 0.4), density=True)
    centers_i = (edges_i[:-1] + edges_i[1:]) / 2
    ancho_i = edges_i[1] - edges_i[0]

    axin = ax.inset_axes(bounds)   # bounds en coordenadas de ejes (0-1)
    m_i = dens_i > 0
    axin.bar(centers_i[m_i], dens_i[m_i], width=ancho_i,
             color="#ff7f0e", alpha=0.9, edgecolor="#d46a00", linewidth=0.4)

    # Ajuste (recta en escala log) solo en modo 0
    if modo == 0 and m_i.sum() >= 3:
        x_i = centers_i[m_i]
        y_i = dens_i[m_i]
        log_y_i = np.log(y_i)
        try:
            popt_i, _ = curve_fit(lambda x, a, b: b - a * x, x_i, log_y_i,
                                  p0=[20.0, np.log(y_i.max())], maxfev=5000)
            a_i, b_i = popt_i
            ss_res = np.sum((log_y_i - (b_i - a_i * x_i)) ** 2)
            ss_tot = np.sum((log_y_i - np.mean(log_y_i)) ** 2)
            r2_i = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            x_curva = np.linspace(inicio_x, 0.4, 200)
            axin.plot(x_curva, np.exp(-a_i * x_curva + b_i),
                      color="#1f77b4", linewidth=1.8)
            axin.text(0.03, 0.08,
                      f"$c\\,e^{{-aA+b}}$\na={a_i:.2f}  R²={r2_i:.3f}",
                      transform=axin.transAxes, fontsize=8 + fs,
                      color="white", ha="left", va="bottom",
                      bbox=dict(facecolor="black", alpha=0.5,
                                edgecolor="none", pad=1.5))
        except RuntimeError:
            pass

    axin.set_yscale("log")
    axin.set_xlim(inicio_x, 0.2)

    # Nombres de ejes como texto dentro del inset (set_xlabel/set_ylabel
    # se recortan al caer fuera del área del inset).
    axin.set_xlabel("")
    axin.set_ylabel("")
    axin.text(0.5, -0.18, "Área relativa", transform=axin.transAxes,
              fontsize=9 + fs, color="white", ha="center", va="top", clip_on=False)
    axin.text(-0.15, 0.5, "Densidad (log)", transform=axin.transAxes,
              fontsize=9 + fs, color="white", ha="center", va="center",
              rotation=90, clip_on=False)

    # Título DENTRO del inset con el tiempo del corte
    tiempo = frame_idx * DT
    unidad = "s" if modo == 0 else "min"
    titulo = f"{tiempo:.2f} {unidad}".replace(".", ",")
    if modo == 0:
        # abajo a la derecha
        axin.text(0.97, 0.08, titulo, transform=axin.transAxes,
                  fontsize=10 + fs, color="white", ha="right", va="bottom",
                  bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5))
    else:
        # arriba a la derecha
        axin.text(0.97, 0.92, titulo, transform=axin.transAxes,
                  fontsize=10 + fs, color="white", ha="right", va="top",
                  bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5))

    # Números, ticks y bordes en blanco para que se lean sobre el fondo oscuro
    axin.tick_params(which="both", labelsize=9 + fs, pad=1,
                     colors="white", labelcolor="white",
                     bottom=True, left=True, top=False, right=False,
                     labelbottom=True, labelleft=True)
    # Sobreescribir si se pidió ocultar
    if not show_xlabel:
        axin.tick_params(axis="x", which="both", labelbottom=False)
    if not show_ylabel:
        axin.tick_params(axis="y", which="both", labelleft=False)
    for spine in axin.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.0)
    axin.patch.set_facecolor("black")
    axin.patch.set_alpha(1.0)   # fondo opaco para tapar ticks del eje principal

# Tres insets pequeños en la zona derecha, en coordenadas de ejes (0-1).
# En modo 1 se ajustan para no solaparse con las líneas de deformación.
if modo == 1:
    # Insets ligeramente más juntos que antes para no tapar las líneas
    # de deformación (t=15 abajo y t=273 arriba).
    _dibujar_inset_hist([0.50, 0.70, 0.46, 0.19], FRAMES_INSET[0],
                        show_xlabel=True, show_ylabel=True)
    _dibujar_inset_hist([0.50, 0.42, 0.46, 0.19], FRAMES_INSET[1],
                        show_xlabel=True, show_ylabel=True)
    _dibujar_inset_hist([0.50, 0.14, 0.46, 0.19], FRAMES_INSET[2],
                        show_xlabel=True, show_ylabel=True)
else:
    _dibujar_inset_hist([0.55, 0.72, 0.42, 0.22], FRAMES_INSET[0])
    _dibujar_inset_hist([0.55, 0.40, 0.42, 0.22], FRAMES_INSET[1])
    _dibujar_inset_hist([0.55, 0.08, 0.42, 0.22], FRAMES_INSET[2])

ax.grid(False)
plt.tight_layout()
fig.savefig(figdir / nombre_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Histograma guardado.")




#  AJUSTE EXPONENCIAL (solo modo 0)


if modo == 0:

    def exponencial(A, a):
        return np.exp(-a * A)

    def r_cuadrado(y_real, y_pred):
        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    coeficientes_a = []
    coeficientes_b = []
    r2_list        = []
    frames_validos = []

    for t, areas_frame in enumerate(areas_por_tiempo):
        areas = np.array(areas_frame)
        areas = areas[areas >= inicio_x]
        if len(areas) < 3:
            continue

        counts, bin_edges = np.histogram(areas, bins=N_BINS_04, range=(inicio_x, 0.4),
                                          density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        mascara = counts > 0
        if mascara.sum() < 3:
            continue

        x = bin_centers[mascara]
        y = counts[mascara]

        # Ajuste lineal en escala log: log(y) = -a*x + b  → recta con intercepto b
        log_y = np.log(y)
        try:
            popt, _ = curve_fit(lambda x, a, b: b - a * x, x, log_y,
                                p0=[20.0, np.log(y.max())], maxfev=2000)
            a_frame, b_frame = popt
            log_y_pred = b_frame - a_frame * x
            r2 = r_cuadrado(log_y, log_y_pred)   # R² en escala log, válido
            coeficientes_a.append(a_frame)
            coeficientes_b.append(b_frame)
            r2_list.append(r2)
            frames_validos.append(t)
        except RuntimeError:
            pass

    coeficientes_a = np.array(coeficientes_a)
    coeficientes_b = np.array(coeficientes_b)
    r2_list        = np.array(r2_list)
    frames_validos = np.array(frames_validos)
    tiempos_validos = frames_validos * DT

    media_a  = np.mean(coeficientes_a)
    std_a    = np.std(coeficientes_a)
    media_b  = np.mean(coeficientes_b)
    std_b    = np.std(coeficientes_b)
    media_r2 = np.mean(r2_list)
    std_r2   = np.std(r2_list)

    print(f"Ajuste exponencial: a = {media_a:.4f} ± {std_a:.4f}")
    print(f"Coeficiente b     : b = {media_b:.4f} ± {std_b:.4f}")
    print(f"R² medio: {media_r2:.4f} ± {std_r2:.4f}")
    print(f"Frames válidos: {len(frames_validos)}/{n_frames}")

    # ── Test KS global ────────────────────────────────────────
    areas_positivas = todas_areas[todas_areas >= inicio_x]
    escala_exp = 1.0 / media_a
    ks_stat, ks_p = kstest(areas_positivas, 'expon', args=(0, escala_exp))
    print(f"Test KS vs exponencial: estadístico={ks_stat:.4f}  p-valor={ks_p:.4f}")


    # ── Figura 1: coeficiente a por tiempo ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempos_validos, coeficientes_a,
            color="#d4b800", marker="o", linestyle="-", linewidth=1,
            markersize=3, alpha=0.8, label="Coeficiente $a$")
    ax.axhline(media_a, color="#00b4b4", linestyle="--", linewidth=1.5,
               label=f"Media = {media_a:.4f}")
    ax.fill_between(tiempos_validos,
                    media_a - std_a, media_a + std_a,
                    color="#00b4b4", alpha=0.15, label=f"±std = {std_a:.4f}")


    ax.set_xlabel(ETIQUETA_TIEMPO)
    ax.set_ylabel("Coeficiente $a$")
    ax.legend(fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figdir / "coeficiente_a100_v7.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


    # ── Figura 2: R² por tiempo con std ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempos_validos, r2_list,
            color="#2e8b57", marker="o", linestyle="-",
            linewidth=1, markersize=3, alpha=0.8)
    ax.axhline(media_r2, color="#2e8b57", linestyle="--", linewidth=1.5,
               label=f"Media R² = {media_r2:.4f}")
    ax.fill_between(tiempos_validos,
                    media_r2 - std_r2, media_r2 + std_r2,
                    color="#2e8b57", alpha=0.15, label=f"±std = {std_r2:.4f}")

    ax.set_ylabel("$R^2$")
    ax.set_xlabel(ETIQUETA_TIEMPO)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(figdir / "calidad_ajuste_v7.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


    # ── Figura 3: histograma global + ajuste directo ─────────
    counts_g, bin_edges_g = np.histogram(areas_positivas, bins=50,
                                          range=(inicio_x, 0.4), density=True)
    bin_centers_g = (bin_edges_g[:-1] + bin_edges_g[1:]) / 2

    mascara_fit = counts_g > 0
    x_fit_g  = bin_centers_g[mascara_fit]
    y_fit_g  = counts_g[mascara_fit]
    log_y_g  = np.log(y_fit_g)
    try:
        popt_g, _ = curve_fit(lambda x, a, b: b - a * x, x_fit_g, log_y_g,
                              p0=[20.0, np.log(y_fit_g.max())], maxfev=5000)
        a_global, b_global = popt_g
        r2_global = r_cuadrado(log_y_g, b_global - a_global * x_fit_g)
    except RuntimeError:
        a_global  = media_a
        b_global  = media_b
        r2_global = float("nan")

    x_fit = np.linspace(inicio_x, 0.4, 300)
    y_fit = np.exp(-a_global * x_fit + b_global)
    print(f"Ajuste directo histograma global: b={b_global:.4f}  a={a_global:.4f}  R²={r2_global:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(bin_centers_g, counts_g, width=bin_edges_g[1] - bin_edges_g[0],
                color="#4c72b0", alpha=0.7, label="Datos")
    axes[0].plot(x_fit, y_fit, color="#d4b800", linewidth=2,
                 label=f"$c\\,e^{{-aA+b}}$  a={a_global:.2f}  R²={r2_global:.3f}")
    axes[0].set_xlabel("Área relativa")
    axes[0].set_ylabel("Densidad")
    axes[0].legend(fontsize=13)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    mascara_pos = counts_g > 0
    axes[1].bar(bin_centers_g[mascara_pos], counts_g[mascara_pos],
                width=bin_edges_g[1] - bin_edges_g[0],
                color="#4c72b0", alpha=0.7, label="Datos")
    axes[1].plot(x_fit, y_fit, color="#d4b800", linewidth=2,
                 label=f"$c\\,e^{{-aA+b}}$  b={b_global:.2f}  a={a_global:.2f}  R²={r2_global:.3f}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Área relativa")
    axes[1].set_ylabel("Densidad (log)")
    axes[1].set_title(f"Distribución global — escala log\nKS: stat={ks_stat:.4f}  p={ks_p:.4f}")
    axes[1].legend(fontsize=13)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(figdir / "ajuste_global_v7.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Figuras guardadas.")