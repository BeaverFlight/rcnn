"""
Конвертер датасета из произвольных папок участков в формат TreeRCNN.

Входная структура (любые имена папок):
    input_dir/
        pp_1_200/
            <anything>.las   (или .laz)
            <anything>.shp   (точки деревьев: поля x, y, z, tree_id)
        pp57-800/
            ...

Выходная структура (готова для NewforDataset):
    output_dir/
        plot_01/
            plot_01.las
            dem.asc
            reference_trees.txt
        plot_02/
            ...

DEM строится из точек класса 2 (ground) LAS-файла.
Если ground-точек нет — используется минимальная высота в каждой ячейке сетки.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("convert_dataset")

DEM_RESOLUTION = 0.5  # метров на пиксель — можно менять
GROUND_CLASS = 2  # LAS classification code для земли


# ---------------------------------------------------------------------------
# LAS / LAZ
# ---------------------------------------------------------------------------


def load_las(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Читает LAS/LAZ файл.

    Returns:
        points: (N, 3) float32 [X, Y, Z] — все точки
        ground_mask: (N,) bool — True для точек класса 2 (ground)
    """
    import laspy

    with laspy.open(str(path)) as f:
        las = f.read()

    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    points = np.column_stack([x, y, z]).astype(np.float32)

    try:
        classification = np.array(las.classification, dtype=np.uint8)
        ground_mask = classification == GROUND_CLASS
    except Exception:
        logger.warning(
            "Не удалось прочитать classification из %s — DEM по минимуму", path.name
        )
        ground_mask = np.zeros(len(points), dtype=bool)

    logger.info(
        "  LAS: %d точек, из них ground: %d (%.1f%%)",
        len(points),
        ground_mask.sum(),
        100.0 * ground_mask.sum() / max(len(points), 1),
    )
    return points, ground_mask


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------


def build_dem(
    points: np.ndarray,
    ground_mask: np.ndarray,
    resolution: float = DEM_RESOLUTION,
) -> tuple[np.ndarray, float, float, float]:
    """
    Строит DEM из облака точек.

    Если есть ground-точки (класс 2) — берёт максимум высот в каждой ячейке
    (ближе к реальной поверхности земли при нормальной плотности съёмки).
    Иначе — берёт минимум по всем точкам (грубое приближение).

    Returns:
        dem:        (rows, cols) float32 — высота земли
        x_orig:     левый нижний угол X
        y_orig:     левый нижний угол Y
        resolution: размер ячейки
    """
    x_min, y_min = float(points[:, 0].min()), float(points[:, 1].min())
    x_max, y_max = float(points[:, 0].max()), float(points[:, 1].max())

    cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    dem = np.full((rows, cols), np.nan, dtype=np.float32)

    src = points[ground_mask] if ground_mask.any() else points
    use_ground = ground_mask.any()

    px = np.floor((src[:, 0] - x_min) / resolution).astype(int)
    py = np.floor((src[:, 1] - y_min) / resolution).astype(int)
    px = np.clip(px, 0, cols - 1)
    py = np.clip(py, 0, rows - 1)

    for i in range(len(src)):
        z = src[i, 2]
        cur = dem[py[i], px[i]]
        if np.isnan(cur):
            dem[py[i], px[i]] = z
        else:
            # ground: берём максимум (верхняя граница земли),
            # fallback (минимум по всем): берём минимум
            dem[py[i], px[i]] = max(cur, z) if use_ground else min(cur, z)

    # Заполняем пустые ячейки ближайшим соседом (простая интерполяция)
    _fill_nan_nearest(dem)

    logger.info(
        "  DEM: %dx%d ячеек, разрешение=%.2f м, источник=%s",
        rows,
        cols,
        resolution,
        "ground class 2" if use_ground else "global minimum",
    )
    return dem, x_min, y_min, resolution


def _fill_nan_nearest(dem: np.ndarray) -> None:
    """
    Заполняет NaN-ячейки DEM методом «ближайший сосед» через scipy.
    In-place операция.
    """
    from scipy.ndimage import distance_transform_edt

    nan_mask = np.isnan(dem)
    if not nan_mask.any():
        return

    # distance_transform_edt возвращает индексы ближайшего не-NaN пикселя
    _, nearest_idx = distance_transform_edt(nan_mask, return_indices=True)
    dem[nan_mask] = dem[nearest_idx[0][nan_mask], nearest_idx[1][nan_mask]]


def save_dem_asc(
    dem: np.ndarray,
    x_orig: float,
    y_orig: float,
    resolution: float,
    path: Path,
) -> None:
    """
    Сохраняет DEM в формате ESRI ASCII Grid (.asc).
    ASC хранит строки сверху вниз, поэтому flipud перед записью.
    """
    rows, cols = dem.shape
    with open(path, "w") as f:
        f.write(f"ncols         {cols}\n")
        f.write(f"nrows         {rows}\n")
        f.write(f"xllcorner     {x_orig:.6f}\n")
        f.write(f"yllcorner     {y_orig:.6f}\n")
        f.write(f"cellsize      {resolution:.6f}\n")
        f.write(f"NODATA_value  -9999\n")
        flipped = np.flipud(dem)
        for row in flipped:
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    logger.info("  DEM сохранён: %s", path)


# ---------------------------------------------------------------------------
# SHP → reference_trees.txt
# ---------------------------------------------------------------------------


def load_shp_trees(path: Path) -> np.ndarray:
    """
    Читает SHP с вершинами деревьев.

    Ожидает поля: x, y, z (регистр не важен).
    tree_id используется только для логирования.

    Returns:
        (M, 3) float32 [x, y, z]
    """
    import geopandas as gpd

    gdf = gpd.read_file(str(path))

    # Нормализуем имена колонок к нижнему регистру
    gdf.columns = [c.lower() for c in gdf.columns]

    missing = [c for c in ("x", "y", "z") if c not in gdf.columns]
    if missing:
        # Пробуем взять координаты из геометрии
        if "geometry" in gdf.columns and not gdf.geometry.is_empty.all():
            logger.info("  Поля %s не найдены — читаем координаты из geometry", missing)
            gdf["x"] = gdf.geometry.x
            gdf["y"] = gdf.geometry.y
            if "z" not in gdf.columns:
                try:
                    gdf["z"] = gdf.geometry.z
                except Exception:
                    raise ValueError(
                        f"Поле 'z' не найдено в {path.name} и недоступно из geometry"
                    )
        else:
            raise ValueError(f"Поля {missing} не найдены в {path.name}")

    trees = gdf[["x", "y", "z"]].to_numpy(dtype=np.float32)

    n_id = gdf["tree_id"].nunique() if "tree_id" in gdf.columns else len(trees)
    logger.info("  SHP: %d деревьев (tree_id уникальных: %d)", len(trees), n_id)
    return trees


def save_reference_trees(trees: np.ndarray, path: Path) -> None:
    """Сохраняет reference_trees.txt: x y z, одно дерево на строку."""
    np.savetxt(str(path), trees, fmt="%.4f", delimiter=" ")
    logger.info("  reference_trees.txt сохранён: %s (%d деревьев)", path, len(trees))


# ---------------------------------------------------------------------------
# Копирование LAS
# ---------------------------------------------------------------------------


def copy_las(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(str(src), str(dst))
    logger.info("  LAS скопирован: %s", dst)


# ---------------------------------------------------------------------------
# Поиск файлов в папке участка
# ---------------------------------------------------------------------------


def find_las(folder: Path) -> Path | None:
    for ext in ("*.las", "*.laz", "*.LAS", "*.LAZ"):
        files = list(folder.glob(ext))
        if files:
            return files[0]
    return None


def find_shp(folder: Path) -> Path | None:
    files = list(folder.glob("*.shp")) + list(folder.glob("*.SHP"))
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Главная логика конвертации
# ---------------------------------------------------------------------------


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    dem_resolution: float = DEM_RESOLUTION,
    start_plot_id: int = 1,
) -> list[int]:
    """
    Обходит все подпапки input_dir, конвертирует каждый участок.

    Returns:
        Список plot_id успешно сконвертированных участков.
    """
    subfolders = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subfolders:
        logger.error("Не найдено подпапок в %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_id = start_plot_id
    successful_ids: list[int] = []

    for folder in subfolders:
        logger.info("Обработка участка: %s → plot_%02d", folder.name, plot_id)

        las_path = find_las(folder)
        shp_path = find_shp(folder)

        if las_path is None:
            logger.warning("  Пропускаем %s: LAS/LAZ файл не найден", folder.name)
            continue
        if shp_path is None:
            logger.warning("  Пропускаем %s: SHP файл не найден", folder.name)
            continue

        try:
            # Создаём выходную папку участка
            plot_dir = output_dir / f"plot_{plot_id:02d}"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # 1. Читаем LAS и строим DEM
            points, ground_mask = load_las(las_path)
            dem, x_orig, y_orig, res = build_dem(points, ground_mask, dem_resolution)

            # 2. Сохраняем DEM
            dem_out = plot_dir / "dem.asc"
            save_dem_asc(dem, x_orig, y_orig, res, dem_out)

            # 3. Читаем SHP и сохраняем reference_trees.txt
            trees = load_shp_trees(shp_path)
            ref_out = plot_dir / "reference_trees.txt"
            save_reference_trees(trees, ref_out)

            # 4. Копируем LAS (сохраняем оригинальный формат .las/.laz)
            las_out = plot_dir / f"plot_{plot_id:02d}{las_path.suffix.lower()}"
            copy_las(las_path, las_out)

            logger.info("  ✓ plot_%02d готов", plot_id)
            successful_ids.append(plot_id)
            plot_id += 1

        except Exception as e:
            logger.error("  Ошибка при обработке %s: %s", folder.name, e, exc_info=True)
            # Продолжаем со следующим участком, не прерывая весь процесс
            continue

    return successful_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Конвертирует папки с участками (LAS + SHP) "
            "в формат TreeRCNN (plot_NN/las + dem.asc + reference_trees.txt)"
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Папка с подпапками участков (pp_1_200, pp57-800, ...)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Куда сохранить конвертированный датасет",
    )
    parser.add_argument(
        "--dem_resolution",
        type=float,
        default=DEM_RESOLUTION,
        help=f"Разрешение DEM в метрах (по умолчанию: {DEM_RESOLUTION})",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="Начальный номер plot_id (по умолчанию: 1)",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="После конвертации вывести блок cross_validation.folds для configs/tree_rcnn.yaml",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error("input_dir не существует: %s", args.input_dir)
        raise SystemExit(1)

    logger.info("Входная папка : %s", args.input_dir)
    logger.info("Выходная папка: %s", args.output_dir)
    logger.info("DEM resolution: %.2f м", args.dem_resolution)

    ids = convert_dataset(
        args.input_dir,
        args.output_dir,
        dem_resolution=args.dem_resolution,
        start_plot_id=args.start_id,
    )

    logger.info("Успешно конвертировано участков: %d", len(ids))
    logger.info("plot_id список: %s", ids)

    if args.print_config and ids:
        _print_folds_config(ids)


def _print_folds_config(ids: list[int], n_folds: int = 4) -> None:
    """
    Выводит готовый YAML-блок cross_validation.folds
    для вставки в configs/tree_rcnn.yaml.
    """
    import math

    chunk = math.ceil(len(ids) / n_folds)
    folds = [ids[i : i + chunk] for i in range(0, len(ids), chunk)]
    # Если фолдов вышло меньше n_folds — не страшно
    actual_folds = len(folds)

    print("\n# ---- Вставь в configs/tree_rcnn.yaml ----")
    print(f"cross_validation:")
    print(f"  n_folds: {actual_folds}")
    print(f"  folds:")
    for fold in folds:
        print(f"    - {fold}")
    print("# ------------------------------------------\n")


if __name__ == "__main__":
    main()
