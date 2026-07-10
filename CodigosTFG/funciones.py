"""
FUNCIONES PARA EL ANÁLISIS DE FRAMES DE SIMULACIÓN


"""

import struct
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from skimage.filters import median
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from skimage import filters, morphology
import hdbscan
import re



#  VALORES POR DEFECTO 
#  Se pueden sobrescribir pasándolos como argumento a las funciones.

N_GRID_DEFAULT     = 70
LIMZMIN_DEFAULT    = -7
LIMZMAX_DEFAULT    =  7
EPSILON_DEFAULT    = 0.0
PERCENTIL_ALTO     = 90
SIGMA              = 0.5
MCS                = 30
MS                 = 30
METHOD             = "eom"


#  LECTURA DE FICHEROS DE LÍMITES (generados por calcular_limites.py)


def leer_limitesxy(simdir):
    """Lee limitesxy.txt. Devuelve un array Nx4 con columnas (xmax xmin ymax ymin).
    En modo 0 tendrá una fila; en modo 1, una por frame."""
    ruta = Path(simdir) / "limitesxy.txt"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}")
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                filas.append([float(v) for v in linea.split()])
    return np.array(filas)


def leer_limitesz(simdir):
    """Lee limitesz.txt. Devuelve (LIMZMIN, LIMZMAX)."""
    ruta = Path(simdir) / "limitesz.txt"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}")
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                partes = linea.split()
                return float(partes[0]), float(partes[1])
    raise ValueError(f"{ruta} está vacío")

#  LECTURA DEL PERCENTIL ÓPTIMO (generado por barrido_percentil.py)

def leer_percentil_bajo(resumepath):
    ruta=Path(resumepath)/"percentil_optimo.txt"
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith("Percentil óptimo"):
                partes = linea.split()
                if len(partes) >= 3:
                    try:
                        return int(partes[2])
                    except ValueError:
                        pass
            


#  LECTURA BINARIA DE LOS FICHEROS DE TRAYECTORIAS (.trr)


def read_frame_bin(filepath, N, frame_offset=0):
    with open(filepath, "rb") as f:
        f.seek(frame_offset)
        raw_header = f.read(72)
        if len(raw_header) < 72:
            raise IOError("No se pudo leer la cabecera del frame")
        header = struct.unpack("<18I", raw_header)
        i_f = header[14]
        t   = struct.unpack("<f", struct.pack("<I", header[16]))[0]
        raw_pos = f.read(N * 12)
        if len(raw_pos) < N * 12:
            raise IOError("No se pudieron leer las posiciones")
        positions = np.frombuffer(raw_pos, dtype=np.float32).reshape(N, 3).copy()
    return {"i_f": i_f, "t": t, "positions": positions}



#  CONSTRUCCIÓN DEL GRID DE LOS MAPAS DE CALOR


def construir_heatmap(positions, x_bins, y_bins, limzmin=None, limzmax=None,
                      n_grid=None):
    """
    Construye un heatmap de n_grid x n_grid celdas contando partículas
    que caigan dentro del rango Z y de cada celda (i, j).

    x_bins e y_bins se calculan en el script principal a partir de los
    límites globales de todos los frames.

    limzmin, limzmax: límites Z. Si no se pasan, se usan los valores por defecto.
    n_grid: número de celdas por lado. Si no se pasa, se usa N_GRID_DEFAULT.
    """
    zmin = limzmin if limzmin is not None else LIMZMIN_DEFAULT
    zmax = limzmax if limzmax is not None else LIMZMAX_DEFAULT
    ng   = n_grid  if n_grid  is not None else N_GRID_DEFAULT
    mask = (positions[:, 2] >= zmin) & (positions[:, 2] < zmax)
    pos  = positions[mask]

    if len(pos) == 0:
        return np.zeros((ng, ng), dtype=int)

    ix = np.clip(np.digitize(pos[:, 0], x_bins) - 1, 0, ng - 1)
    iy = np.clip(np.digitize(pos[:, 1], y_bins) - 1, 0, ng - 1)

    heatmap = np.zeros((ng, ng), dtype=int)
    np.add.at(heatmap, (ix, iy), 1)

    return heatmap.T   # transponer: filas=Y, columnas=X (convención imagen)



#  SUAVIZADO Y PREPROCESADO



def aplicar_histeresis(suave_mg, percentil_bajo, percentil_alto=PERCENTIL_ALTO):
    """Histéresis + opening + remove_small_objects. Devuelve valores enteros."""
    if suave_mg[suave_mg > 0].size == 0:
        return np.zeros_like(suave_mg, dtype=int)
    low  = np.percentile(suave_mg, percentil_bajo)
    high = np.percentile(suave_mg, percentil_alto)
    mascara        = filters.apply_hysteresis_threshold(suave_mg, low, high)
    mascara_limpia = morphology.opening(mascara, morphology.disk(3))
    segmentacion   = morphology.remove_small_objects(mascara_limpia, max_size=25)
    return np.clip(np.round(suave_mg * segmentacion), 0, None).astype(int)



def filtromedian(heatmap):

    suave = median(heatmap.astype(np.uint16), disk(2))

    return suave


#CONSTRUCCIÓN DE LA LISTA DE POSICIONES PARA PASARLE A HDBSCAN

def construir_lista(suave):
    i, j = np.indices(suave.shape)
    return np.repeat(
        np.stack((i.flatten(), j.flatten()), axis=1),
        suave.flatten(),
        axis=0,
    )



#  CLUSTERING HDBSCAN


def get_labels(posiciones, min_cluster_size, min_samples, method,
               epsilon=None, n_grid=None):
    eps = epsilon if epsilon is not None else EPSILON_DEFAULT
    ng  = n_grid  if n_grid  is not None else N_GRID_DEFAULT
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size          = min_cluster_size,
        min_samples               = min_samples,
        cluster_selection_epsilon = eps,
        metric                    = "euclidean",
        cluster_selection_method  = method,
        #max_cluster_size =2000 

    )
    clusterer.fit(posiciones)
    labels = clusterer.labels_ + 1   # ruido → 0

    matriz_vis = np.zeros((ng, ng))
    if len(labels) > 0:
        mat = np.column_stack((posiciones, labels))
        mat = np.unique(mat, axis=0)
        matriz_vis[mat[:, 0].astype(int), mat[:, 1].astype(int)] = mat[:, 2]

    return matriz_vis.astype(int)



#  CÓDIGO PARA ELIMINAR RUIDO DE LOS LABELS Y CALCULAR AMI 
#  SOLO PARA LOS LABELS QUE NO SON RUIDO (0)


def filtro_sin_ruido(y_true, y_pred):
    mask = (y_true != 0) | (y_pred != 0)
    return y_true[mask], y_pred[mask]


#AJUSTE LINEAL DE DATOS (para calcular la pendiente de AMI vs lag)


def ajuste_lineal(datos):
    datos = np.array(datos, dtype=float)
    datos = datos[~np.isnan(datos)]
    if len(datos) < 2:
        return 0.0, float(np.nanmean(datos)), np.zeros((2, 2))
    x = np.arange(len(datos))
    params, cov = curve_fit(lambda x, a, b: a * x + b, x, datos)
    return params[0], params[1], cov



