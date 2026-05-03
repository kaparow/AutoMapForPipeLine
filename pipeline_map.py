import numpy as np
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go

# ============================================================
# 1. ТРАССА ТРУБОПРОВОДА
# Прямолинейные участки с поворотами 90°,
# соединённые плавными дугами
# ============================================================


def arc(x0, y0, heading_from, heading_to, r=20, n=35):
    """Строит дугу поворота из точки (x0, y0)."""
    delta = (heading_to - heading_from + np.pi) % (2 * np.pi) - np.pi
    perp = heading_from + np.pi / 2 if delta > 0 else heading_from - np.pi / 2

    cx = x0 + r * np.cos(perp)
    cy = y0 + r * np.sin(perp)

    a_start = np.arctan2(y0 - cy, x0 - cx)
    angles = np.linspace(a_start, a_start + delta, n)

    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return x, y


def build_pipeline_route():
    """Удлинённая трасса с 4 поворотами и большим числом точек."""
    R = 20
    segments_x, segments_y = [], []

    E = 0.0
    N = np.pi / 2
    W = np.pi
    S = -np.pi / 2

    # 1) Восток 200 м
    t = np.linspace(0, 200, 120)
    segments_x.append(t)
    segments_y.append(np.zeros_like(t))

    # Поворот 1: восток -> север
    ax, ay = arc(200, 0, E, N, r=R)
    segments_x.append(ax)
    segments_y.append(ay)

    # 2) Север 140 м
    x0, y0 = ax[-1], ay[-1]
    t = np.linspace(0, 140, 90)
    segments_x.append(np.full_like(t, x0))
    segments_y.append(y0 + t)

    # Поворот 2: север -> восток
    x0, y0 = segments_x[-1][-1], segments_y[-1][-1]
    ax, ay = arc(x0, y0, N, E, r=R)
    segments_x.append(ax)
    segments_y.append(ay)

    # 3) Восток 240 м
    x0, y0 = ax[-1], ay[-1]
    t = np.linspace(0, 240, 150)
    segments_x.append(x0 + t)
    segments_y.append(np.full_like(t, y0))

    # Поворот 3: восток -> юг
    x0, y0 = segments_x[-1][-1], segments_y[-1][-1]
    ax, ay = arc(x0, y0, E, S, r=R)
    segments_x.append(ax)
    segments_y.append(ay)

    # 4) Юг 170 м
    x0, y0 = ax[-1], ay[-1]
    t = np.linspace(0, 170, 105)
    segments_x.append(np.full_like(t, x0))
    segments_y.append(y0 - t)

    # Поворот 4: юг -> запад
    x0, y0 = segments_x[-1][-1], segments_y[-1][-1]
    ax, ay = arc(x0, y0, S, W, r=R)
    segments_x.append(ax)
    segments_y.append(ay)

    # 5) Запад 120 м
    x0, y0 = ax[-1], ay[-1]
    t = np.linspace(0, 120, 80)
    segments_x.append(x0 - t)
    segments_y.append(np.full_like(t, y0))

    x_pipe = np.concatenate(segments_x)
    y_pipe = np.concatenate(segments_y)
    return x_pipe, y_pipe


# ============================================================
# 2. УСЛОВНЫЙ РЕЛЬЕФ ДЛЯ ВИЗУАЛИЗАЦИИ
# ============================================================


def build_terrain(x_pipe, y_pipe):
    pad = 45
    xg = np.linspace(x_pipe.min() - pad, x_pipe.max() + pad, 150)
    yg = np.linspace(y_pipe.min() - pad, y_pipe.max() + pad, 140)
    Xg, Yg = np.meshgrid(xg, yg)

    # Рельеф условный и служит только для визуализации (не измерения робота).
    Zg = (
        0.9 * np.sin(Xg / 90)
        + 0.6 * np.cos(Yg / 75)
        + 0.45 * np.sin((Xg + 0.5 * Yg) / 120)
    )

    # Локальные, но плавные формы местности.
    Zg += 1.8 * np.exp(-((Xg - 260) ** 2 + (Yg - 120) ** 2) / 19000)
    Zg -= 1.2 * np.exp(-((Xg - 430) ** 2 + (Yg - 40) ** 2) / 12000)
    return xg, yg, Zg


# ============================================================
# 3. ВЫСОТНЫЙ ПРОФИЛЬ ТРУБОПРОВОДА
# ============================================================


def build_pipe_height(x_pipe, y_pipe, xg, yg, Zg):
    interp = RegularGridInterpolator((yg, xg), Zg, bounds_error=False, fill_value=0.0)
    z_ground = interp(np.column_stack([y_pipe, x_pipe]))

    n = len(x_pipe)
    s = np.linspace(0, 1, n)

    # Базовый проектный просвет с мягкими изменениями на длинных прямых участках.
    clearance = 2.6 + 0.25 * np.sin(2 * np.pi * s) + 0.18 * np.sin(7 * np.pi * s)

    # Основное высокое препятствие (подъём, верхняя полка, спуск — почти вертикальные фронты).
    x_up_start, x_up_end = 410, 416
    x_top_end, x_down_end = 450, 456

    mask_up = (x_pipe >= x_up_start) & (x_pipe < x_up_end)
    mask_top = (x_pipe >= x_up_end) & (x_pipe <= x_top_end)
    mask_down = (x_pipe > x_top_end) & (x_pipe <= x_down_end)

    if mask_up.any():
        clearance[mask_up] = np.linspace(3.0, 12.0, mask_up.sum())
    if mask_top.any():
        clearance[mask_top] = 12.0
    if mask_down.any():
        clearance[mask_down] = np.linspace(12.0, 3.1, mask_down.sum())

    # Несколько низких препятствий в разных местах.
    small_obstacles = [
        (155, 166, 5.2),
        (305, 315, 5.0),
        (520, 532, 5.4),
    ]

    for x1, x2, peak in small_obstacles:
        mask = (x_pipe >= x1) & (x_pipe <= x2)
        if mask.any():
            idx = np.where(mask)[0]
            t = np.linspace(0, 1, len(idx))
            hump = 3.0 + (peak - 3.0) * np.sin(np.pi * t)
            clearance[idx] = np.maximum(clearance[idx], hump)

    z_pipe = z_ground + clearance
    return z_pipe, z_ground


# ============================================================
# 4. ОПОРЫ
# ============================================================


def place_supports(x_pipe, y_pipe, z_pipe, z_ground, base_step=12):
    n = len(x_pipe)
    supports = [0]

    for i in range(base_step, n - 1, base_step):
        supports.append(i)

    # Уплотнение опор в зонах поворотов и препятствий/крутых профилей.
    turns = detect_turns(x_pipe, y_pipe, angle_threshold_deg=40, window=10, min_gap=35)
    steep = np.where(np.abs(np.gradient(z_pipe)) > 0.65)[0]

    extra = []
    for center in turns + steep.tolist()[::8]:
        for d in (-6, 0, 6):
            j = center + d
            if 0 < j < n - 1:
                extra.append(j)

    supports = sorted(set(supports + extra + [n - 1]))
    return supports


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

support_indices = place_supports(x_pipe, y_pipe, z_pipe, z_ground_under_pipe, base_step=12)

n_total = len(x_pipe)

turn_indices = detect_turns(x_pipe, y_pipe, angle_threshold_deg=40, window=10, min_gap=35)
obstacle_indices = detect_obstacles(z_pipe, z_ground_under_pipe, clearance_threshold=4.5)

char_points = {'Начало трассы': 0, 'Конец трассы': n_total - 1}
for i, idx in enumerate(turn_indices[:6]):
    char_points[f'Поворот {i+1}'] = idx
for i, idx in enumerate(obstacle_indices[:8]):
    char_points[f'Препятствие {i+1}'] = idx

print(f"Точек трассы:     {n_total}")
print(f"Опор:             {len(support_indices)}")
print(f"Поворотов:        {len(turn_indices)}")
print(f"Препятствий:      {len(obstacle_indices)}")
print(f"X: {x_pipe.min():.0f} → {x_pipe.max():.0f} м")
print(f"Y: {y_pipe.min():.0f} → {y_pipe.max():.0f} м")
print(f"Мин. отметка грунта: {z_ground_under_pipe.min():.2f} м")
print(f"Макс. отметка грунта: {z_ground_under_pipe.max():.2f} м")
print(f"Макс. просвет:    {(z_pipe - z_ground_under_pipe).max():.1f} м")


# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ PLOTLY
# ============================================================

fig = go.Figure()

fig.add_trace(go.Surface(
    x=xg,
    y=yg,
    z=Zg,
    colorscale='YlGn',
    opacity=0.82,
    showscale=False,
    name='Условная поверхность земли'
))

fig.add_trace(go.Scatter3d(
    x=x_pipe,
    y=y_pipe,
    z=z_ground_under_pipe,
    mode='lines',
    line=dict(color='rgba(180,40,20,0.55)', width=4, dash='dash'),
    name='Проекция оси'
))

fig.add_trace(go.Scatter3d(
    x=x_pipe,
    y=y_pipe,
    z=z_pipe,
    mode='lines',
    line=dict(color='red', width=8),
    name='Ось трубопровода'
))

x_sup_lines, y_sup_lines, z_sup_lines = [], [], []
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

start_end_labels, start_end_x, start_end_y, start_end_z = [], [], [], []
special_labels, special_x, special_y, special_z = [], [], [], []

for label, idx in char_points.items():
    if 'Поворот' in label or 'Препятствие' in label:
        special_labels.append(label)
        special_x.append(x_pipe[idx])
        special_y.append(y_pipe[idx])
        special_z.append(z_pipe[idx] + 0.8)
    else:
        start_end_labels.append(label)
        start_end_x.append(x_pipe[idx])
        start_end_y.append(y_pipe[idx])
        start_end_z.append(z_pipe[idx] + 0.8)

fig.add_trace(go.Scatter3d(
    x=start_end_x,
    y=start_end_y,
    z=start_end_z,
    mode='markers+text',
    marker=dict(size=7, color='royalblue', symbol='circle'),
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

fig.update_layout(
    title='Рисунок 1.3 — Трассировочная карта надземного трубопровода',
    scene=dict(
        xaxis_title='X, м',
        yaxis_title='Y, м',
        zaxis_title='Высота, м',
        aspectratio=dict(x=3.2, y=1.6, z=0.8),
        camera=dict(eye=dict(x=1.7, y=-1.8, z=1.0))
    ),
    legend=dict(x=0.01, y=0.98, bgcolor='rgba(255,255,255,0.85)'),
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.show()
