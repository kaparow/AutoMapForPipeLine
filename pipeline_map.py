import numpy as np
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go

# ============================================================
# 1. ТРАССА ТРУБОПРОВОДА
# Прямолинейные участки с поворотами строго 90°,
# соединённые плавными дугами
# ============================================================


def arc(x0, y0, heading_from, heading_to, r=18, n=25):
    """
    Строит дугу поворота из точки (x0, y0).
    heading_from, heading_to — углы направления в радианах.
    Поворот по кратчайшей дуге.
    """
    delta = (heading_to - heading_from + np.pi) % (2 * np.pi) - np.pi

    if delta > 0:  # левый поворот
        perp = heading_from + np.pi / 2
    else:  # правый поворот
        perp = heading_from - np.pi / 2

    cx = x0 + r * np.cos(perp)
    cy = y0 + r * np.sin(perp)

    a_start = np.arctan2(y0 - cy, x0 - cx)
    a_end = a_start + delta

    angles = np.linspace(a_start, a_end, n)
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return x, y


def build_pipeline_route():
    """
    Трасса из 4 прямолинейных участков с тремя поворотами по 90°.
    Направления: восток → север → восток → юг
    """
    R = 18
    segments_x = []
    segments_y = []

    E = 0.0
    N = np.pi / 2
    S = -np.pi / 2

    # --- Участок 1: на восток 120 м ---
    t = np.linspace(0, 120, 60)
    segments_x.append(t)
    segments_y.append(np.zeros(60))

    # --- Поворот 1: восток → север ---
    ax, ay = arc(120, 0, E, N, r=R)
    segments_x.append(ax)
    segments_y.append(ay)

    # --- Участок 2: на север 80 м ---
    x0, y0 = ax[-1], ay[-1]
    t = np.linspace(0, 80, 45)
    segments_x.append(np.full(45, x0))
    segments_y.append(y0 + t)

    # --- Поворот 2: север → восток ---
    x0, y0 = segments_x[-1][-1], segments_y[-1][-1]
    ax, ay = arc(x0, y0, N, E, r=R)
    segments_x.append(ax)
    segments_y.append(ay)

    # --- Участок 3: на восток 100 м (через овраг) ---
    x0, y0 = ax[-1], ay[-1]
    t = np.linspace(0, 100, 55)
    segments_x.append(x0 + t)
    segments_y.append(np.full(55, y0))

    # --- Поворот 3: восток → юг ---
    x0, y0 = segments_x[-1][-1], segments_y[-1][-1]
    ax, ay = arc(x0, y0, E, S, r=R)
    segments_x.append(ax)
    segments_y.append(ay)

    # --- Участок 4: на юг 70 м ---
    x0, y0 = ax[-1], ay[-1]
    t = np.linspace(0, 70, 40)
    segments_x.append(np.full(40, x0))
    segments_y.append(y0 - t)

    x_pipe = np.concatenate(segments_x)
    y_pipe = np.concatenate(segments_y)
    return x_pipe, y_pipe


# ============================================================
# 2. УСЛОВНАЯ ПОВЕРХНОСТЬ ЗЕМЛИ
# ============================================================


def build_terrain(x_pipe, y_pipe):
    pad = 35
    xg = np.linspace(x_pipe.min() - pad, x_pipe.max() + pad, 100)
    yg = np.linspace(y_pipe.min() - pad, y_pipe.max() + pad, 100)
    Xg, Yg = np.meshgrid(xg, yg)

    # Условная ровная плоскость для визуализации (не результат измерений робота)
    Zg = np.zeros_like(Xg)

    return xg, yg, Zg


# ============================================================
# 3. ВЫСОТНЫЙ ПРОФИЛЬ ТРУБОПРОВОДА
# ============================================================


def build_pipe_height(x_pipe, y_pipe, xg, yg, Zg):
    # Земля в демонстрационной модели принимается ровной: z = 0
    z_ground = np.zeros(len(x_pipe))

    clearance = np.full(len(x_pipe), 2.5)

    # Подъём трубы над условной зоной препятствия
    mask_up = (x_pipe > 200) & (x_pipe <= 215)
    mask_down = (x_pipe > 215) & (x_pipe < 232)

    if mask_up.sum() > 0:
        clearance[mask_up] = np.linspace(2.5, 9.0, mask_up.sum())
    if mask_down.sum() > 0:
        clearance[mask_down] = np.linspace(9.0, 2.5, mask_down.sum())

    z_pipe = z_ground + clearance
    return z_pipe, z_ground


# ============================================================
# 4. ОПОРЫ
# ============================================================


def place_supports(x_pipe, y_pipe, z_pipe, z_ground, step=10):
    n = len(x_pipe)
    indices = []

    for i in range(0, n, step):
        # В зоне оврага только крайние опоры
        if 202 < x_pipe[i] < 230:
            continue
        indices.append(i)

    idx_ovr_start = int(np.argmin(np.abs(x_pipe - 202)))
    idx_ovr_end = int(np.argmin(np.abs(x_pipe - 230)))

    indices = sorted(set(indices + [0, idx_ovr_start, idx_ovr_end, n - 1]))
    return indices


# ============================================================
# 5. АВТОМАТИЧЕСКОЕ ОБНАРУЖЕНИЕ ХАРАКТЕРНЫХ ТОЧЕК
# ============================================================


def detect_turns(x, y, angle_threshold_deg=60, window=12, min_gap=20):
    turns = []

    for i in range(window, len(x) - window):
        v1 = np.array([x[i] - x[i - window], y[i] - y[i - window]])
        v2 = np.array([x[i + window] - x[i], y[i + window] - y[i]])

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-6 or n2 < 1e-6:
            continue

        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.degrees(np.arccos(cos_a))

        if angle > angle_threshold_deg:
            if not turns or (i - turns[-1]) > min_gap:
                turns.append(i)

    return turns


def detect_obstacles(z_pipe, z_ground, clearance_threshold=4.5):
    clearance = z_pipe - z_ground
    obstacles = []
    in_obstacle = False

    for i in range(len(clearance)):
        if clearance[i] > clearance_threshold and not in_obstacle:
            obstacles.append(i)
            in_obstacle = True
        elif clearance[i] <= clearance_threshold:
            in_obstacle = False

    return obstacles


# ============================================================
# 6. СБОРКА ДАННЫХ
# ============================================================

x_pipe, y_pipe = build_pipeline_route()
xg, yg, Zg = build_terrain(x_pipe, y_pipe)
z_pipe, z_ground_under_pipe = build_pipe_height(x_pipe, y_pipe, xg, yg, Zg)

support_indices = place_supports(
    x_pipe, y_pipe, z_pipe, z_ground_under_pipe, step=10
)

n_total = len(x_pipe)

turn_indices = detect_turns(x_pipe, y_pipe, angle_threshold_deg=40)
obstacle_indices = detect_obstacles(
    z_pipe, z_ground_under_pipe, clearance_threshold=4.5
)

char_points = {'Начало трассы': 0, 'Конец трассы': n_total - 1}
for i, idx in enumerate(turn_indices):
    char_points[f'Поворот {i+1}'] = idx
for i, idx in enumerate(obstacle_indices):
    char_points[f'Препятствие {i+1}'] = idx

print(f"Точек трассы:     {n_total}")
print(f"Опор:             {len(support_indices)}")
print(f"Поворотов:        {len(turn_indices)}")
print(f"Препятствий:      {len(obstacle_indices)}")
print(f"X: {x_pipe.min():.0f} → {x_pipe.max():.0f} м")
print(f"Y: {y_pipe.min():.0f} → {y_pipe.max():.0f} м")
print(f"Макс. просвет:    {(z_pipe - z_ground_under_pipe).max():.1f} м")


# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ PLOTLY
# ============================================================

fig = go.Figure()

# --- Условная поверхность земли ---
fig.add_trace(go.Surface(
    x=xg,
    y=yg,
    z=Zg,
    colorscale='Greens',
    opacity=0.78,
    showscale=False,
    name='Условная плоскость z=0'
))

# --- Проекция оси трубопровода на условную поверхность ---
fig.add_trace(go.Scatter3d(
    x=x_pipe,
    y=y_pipe,
    z=z_ground_under_pipe,
    mode='lines',
    line=dict(color='rgba(180,40,20,0.55)', width=4, dash='dash'),
    name='Проекция оси'
))

# --- Ось трубопровода ---
fig.add_trace(go.Scatter3d(
    x=x_pipe,
    y=y_pipe,
    z=z_pipe,
    mode='lines',
    line=dict(color='red', width=8),
    name='Ось трубопровода'
))

# --- Опоры ---
x_sup_lines = []
y_sup_lines = []
z_sup_lines = []

for idx in support_indices:
    x_sup_lines += [x_pipe[idx], x_pipe[idx], None]
    y_sup_lines += [y_pipe[idx], y_pipe[idx], None]
    z_sup_lines += [z_ground_under_pipe[idx], z_pipe[idx], None]

fig.add_trace(go.Scatter3d(
    x=x_sup_lines,
    y=y_sup_lines,
    z=z_sup_lines,
    mode='lines',
    line=dict(color='saddlebrown', width=5),
    name='Опоры'
))

# --- Начало и конец трассы ---
start_end_labels = []
start_end_x = []
start_end_y = []
start_end_z = []

# --- Повороты и препятствия ---
special_labels = []
special_x = []
special_y = []
special_z = []

for label, idx in char_points.items():
    if 'Поворот' in label or 'Препятствие' in label:
        special_labels.append(label)
        special_x.append(x_pipe[idx])
        special_y.append(y_pipe[idx])
        special_z.append(z_pipe[idx] + 0.6)
    else:
        start_end_labels.append(label)
        start_end_x.append(x_pipe[idx])
        start_end_y.append(y_pipe[idx])
        start_end_z.append(z_pipe[idx] + 0.6)

fig.add_trace(go.Scatter3d(
    x=start_end_x,
    y=start_end_y,
    z=start_end_z,
    mode='markers+text',
    marker=dict(size=6, color='royalblue', symbol='circle'),
    text=start_end_labels,
    textposition='top center',
    name='Начало / конец'
))

fig.add_trace(go.Scatter3d(
    x=special_x,
    y=special_y,
    z=special_z,
    mode='markers+text',
    marker=dict(size=6, color='darkorange', symbol='diamond'),
    text=special_labels,
    textposition='top center',
    name='Повороты / препятствия'
))

# --- Настройка отображения ---
fig.update_layout(
    title='Рисунок 1.3 — Трассировочная карта надземного трубопровода',
    scene=dict(
        xaxis_title='X, м',
        yaxis_title='Y, м',
        zaxis_title='Высота, м',
        aspectratio=dict(x=2.8, y=1.2, z=0.5),
        camera=dict(
            eye=dict(x=1.6, y=-1.7, z=0.9)
        )
    ),
    legend=dict(
        x=0.01,
        y=0.98,
        bgcolor='rgba(255,255,255,0.85)'
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.show()
