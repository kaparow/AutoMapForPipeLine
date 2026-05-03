import numpy as np
import plotly.graph_objects as go


def build_route():
    """Extended route preserving start/end and characteristic turns."""
    route = np.array([
        [0, 0],
        [50, 0],
        [110, 20],   # Поворот 1
        [180, 20],
        [250, 55],   # Поворот 2
        [340, 55],
        [420, 15],   # Поворот 3
        [520, 15],
        [620, 45],
        [730, 45],
        [860, 0],    # Конец трассы
    ], dtype=float)
    return route


def sample_polyline(points: np.ndarray, step: float = 8.0):
    sampled = [points[0]]
    chainage = [0.0]

    for i in range(1, len(points)):
        p0, p1 = points[i - 1], points[i]
        vec = p1 - p0
        dist = np.linalg.norm(vec)
        n = max(1, int(dist // step))
        for k in range(1, n + 1):
            t = k / n
            p = p0 + t * vec
            sampled.append(p)
            chainage.append(chainage[-1] + np.linalg.norm(sampled[-1] - sampled[-2]))

    return np.array(sampled), np.array(chainage)


def terrain_elevation(x, y, s):
    """Smooth, realistic undulating terrain without chaotic noise."""
    base = 12.0
    regional = 2.8 * np.sin(x / 180.0) + 2.0 * np.cos(y / 150.0)
    mid_scale = 1.2 * np.sin((x + y) / 95.0) + 0.8 * np.cos((x - 0.7 * y) / 80.0)
    long_trend = 0.004 * s
    return base + regional + mid_scale + long_trend


def pipeline_profile(ground_z, s):
    """Pipeline profile with mild project-like grade changes + explicit steep section."""
    clearance = 4.5
    pipe_z = ground_z + clearance

    # Slight adjustments on long straights (not snake-like)
    pipe_z += 0.45 * np.sin(s / 120.0) + 0.25 * np.cos(s / 85.0)

    # Steep lift -> top horizontal -> steep drop segment
    s_up_start = 430.0
    s_up_end = 448.0
    s_top_end = 525.0
    s_down_end = 543.0
    lift_h = 15.0

    lift = np.zeros_like(s)

    up_mask = (s >= s_up_start) & (s <= s_up_end)
    top_mask = (s > s_up_end) & (s <= s_top_end)
    down_mask = (s > s_top_end) & (s <= s_down_end)

    lift[up_mask] = lift_h * (s[up_mask] - s_up_start) / (s_up_end - s_up_start)
    lift[top_mask] = lift_h
    lift[down_mask] = lift_h * (1 - (s[down_mask] - s_top_end) / (s_down_end - s_top_end))

    pipe_z += lift
    return pipe_z


def support_indices(points: np.ndarray, s: np.ndarray, turns_idx: list[int]):
    """Mostly uniform supports, slightly denser near turns/steep profile."""
    target_step = 22.0
    indices = [0]
    next_s = target_step

    while next_s < s[-1]:
        i = int(np.argmin(np.abs(s - next_s)))
        indices.append(i)

        # adaptive local step near route turns/steep section
        near_turn = any(abs(i - t) < 4 for t in turns_idx)
        in_steep = 425 <= s[i] <= 550

        local_step = target_step
        if near_turn:
            local_step = 17.0
        elif in_steep:
            local_step = 15.0

        next_s += local_step

    if indices[-1] != len(points) - 1:
        indices.append(len(points) - 1)

    return sorted(set(indices))


def create_figure():
    route = build_route()
    points, s = sample_polyline(route, step=8)
    x, y = points[:, 0], points[:, 1]

    ground_z = terrain_elevation(x, y, s)
    pipe_z = pipeline_profile(ground_z, s)

    # Build surrounding terrain surface
    gx = np.linspace(-20, 900, 70)
    gy = np.linspace(-40, 110, 45)
    GX, GY = np.meshgrid(gx, gy)
    GS = np.sqrt((GX - GX.min()) ** 2 + (GY - GY.min()) ** 2)
    GZ = terrain_elevation(GX, GY, GS)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=GX,
        y=GY,
        z=GZ,
        colorscale="Earth",
        opacity=0.85,
        showscale=False,
        name="Рельеф"
    ))

    # Pipeline
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=pipe_z,
        mode="lines",
        line=dict(color="crimson", width=7),
        name="Трубопровод"
    ))

    # Ground projection (for readability)
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=ground_z,
        mode="lines",
        line=dict(color="sienna", width=3, dash="dot"),
        name="Трасса по земле"
    ))

    # Turn labels
    turn_points = [2, 4, 6]
    turn_names = ["Поворот 1", "Поворот 2", "Поворот 3"]
    for idx, name in zip(turn_points, turn_names):
        tx, ty = route[idx]
        ti = int(np.argmin((x - tx) ** 2 + (y - ty) ** 2))
        fig.add_trace(go.Scatter3d(
            x=[x[ti]], y=[y[ti]], z=[pipe_z[ti] + 1.5],
            mode="markers+text",
            marker=dict(size=5, color="navy"),
            text=[name],
            textposition="top center",
            name=name,
            showlegend=False
        ))

    # Start and end markers
    for i, lbl, col in [(0, "Начало трассы", "green"), (-1, "Конец трассы", "red")]:
        fig.add_trace(go.Scatter3d(
            x=[x[i]], y=[y[i]], z=[pipe_z[i] + 1.8],
            mode="markers+text",
            marker=dict(size=7, color=col),
            text=[lbl],
            textposition="top center",
            name=lbl
        ))

    # Supports
    turn_sample_idx = [int(np.argmin((x - route[i, 0]) ** 2 + (y - route[i, 1]) ** 2)) for i in turn_points]
    sup_idx = support_indices(points, s, turn_sample_idx)
    for i in sup_idx:
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[i]],
            y=[y[i], y[i]],
            z=[ground_z[i], pipe_z[i]],
            mode="lines",
            line=dict(color="gray", width=3),
            name="Опора" if i == sup_idx[0] else None,
            showlegend=(i == sup_idx[0])
        ))

    # Main obstacle + several lower obstacles
    obstacles = [
        (300, 44, 13, "Основное препятствие"),
        (145, 7, 6.0, "Препятствие 1"),
        (490, 26, 7.5, "Препятствие 2"),
        (690, 38, 5.5, "Препятствие 3"),
        (805, 6, 6.8, "Препятствие 4"),
    ]

    for ox, oy, h, name in obstacles:
        base = terrain_elevation(np.array([ox]), np.array([oy]), np.array([0]))[0]
        fig.add_trace(go.Scatter3d(
            x=[ox, ox], y=[oy, oy], z=[base, base + h],
            mode="lines+text",
            line=dict(color="black", width=6),
            text=[None, name],
            textposition="top center",
            name=name
        ))

    fig.update_layout(
        title="3D-профиль трубопровода: удлиненная трасса, рельеф, опоры и препятствия",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Высота",
            aspectmode="manual",
            aspectratio=dict(x=2.6, y=1.0, z=0.7),
            camera=dict(eye=dict(x=1.8, y=1.4, z=0.8)),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


if __name__ == "__main__":
    figure = create_figure()
    figure.write_html("pipeline_3d.html", include_plotlyjs="cdn")
    print("Saved: pipeline_3d.html")
