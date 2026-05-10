"use strict";

const mapElement = document.getElementById("map");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resetBtn = document.getElementById("resetBtn");
const finishBtn = document.getElementById("finishBtn");
const rotateBtn = document.getElementById("rotateBtn");
const speedSlider = document.getElementById("speedSlider");
const speedValue = document.getElementById("speedValue");
const statusText = document.getElementById("statusText");
const pointMetric = document.getElementById("pointMetric");
const distanceMetric = document.getElementById("distanceMetric");
const picketMetric = document.getElementById("picketMetric");
const progressFill = document.getElementById("progressFill");

const OBSTACLE_CLEARANCE_THRESHOLD_M = 4.3;
const ROBOT_POINT_COUNT = 3000;
const ROBOT_OBSTACLE_WINDOWS = Object.freeze([
  { start: 132, end: 141, rise: 2.2 },
  { start: 254, end: 268, rise: 2.5 },
  { start: 366, end: 378, rise: 2.1 },
  { start: 493, end: 512, rise: 2.7 },
  { start: 621, end: 639, rise: 2.35 },
  { start: 725, end: 744, rise: 2.6 }
]);

const state = {
  playing: false,
  autoRotate: false,
  playTimer: null,
  rotateTimer: null,
  cameraAngle: -0.82
};

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function linspace(start, stop, count) {
  if (count <= 1) {
    return [start];
  }

  const step = (stop - start) / (count - 1);
  return Array.from({ length: count }, (_, i) => start + step * i);
}

function zeros(count) {
  return Array.from({ length: count }, () => 0);
}

// ROBOT_POINTS имитирует поток координат от робота. В реальной системе вместо
// этого демонстрационного массива будут использоваться координаты, полученные
// от GPS/IMU/одометрии/алгоритма сопровождения трубопровода.
const ROBOT_POINTS = Object.freeze(Array.from({ length: ROBOT_POINT_COUNT }, (_, i) => {
  const timeStepS = 2;
  const tail = Math.max(0, i - 430);
  const tailBendX = tail === 0
    ? 0
    : 18 * Math.sin(tail / 52 + 0.4) + 8 * Math.sin(tail / 27);
  const tailBendY = tail === 0
    ? 0
    : 50 * Math.sin(tail / 58) - 20 * Math.sin(tail / 24) + 0.1 * tail * Math.sin(tail / 170);
  const x =
    i * 2.05 +
    13.5 * Math.sin(i / 39) +
    4.8 * Math.sin(i / 13) +
    tailBendX;
  const y =
    58 * Math.sin(i / 76) +
    22 * Math.sin(i / 34 + 0.6) +
    0.05 * i +
    tailBendY;
  const groundZ =
    0.34 * Math.sin(x / 165) +
    0.22 * Math.cos(y / 88) +
    0.08 * Math.sin((x + y) / 95) +
    0.06 * Math.cos(i / 49);
  const obstacleWindow = ROBOT_OBSTACLE_WINDOWS.find((window) => (
    i >= window.start && i <= window.end
  ));
  const isRobotObstacle = obstacleWindow !== undefined;

  const localObstacleRise = isRobotObstacle
    ? obstacleWindow.rise * Math.sin(Math.PI * ((i - obstacleWindow.start) / (obstacleWindow.end - obstacleWindow.start)))
    : 0;

  const clearance =
    2.78 +
    0.18 * Math.sin(i / 31) +
    0.11 * Math.cos(i / 17) +
    localObstacleRise;

  const point = {
    x: round(x),
    y: round(y),
    z: round(groundZ + clearance),
    groundZ: round(groundZ),
    timestamp: new Date(Date.UTC(2026, 4, 10, 7, 0, i * timeStepS)).toISOString()
  };

  if (isRobotObstacle) {
    point.obstacle = true;
  }

  return point;
}));

function cumulativeDistance(x, y) {
  const result = [0];
  for (let i = 1; i < x.length; i += 1) {
    const dx = x[i] - x[i - 1];
    const dy = y[i] - y[i - 1];
    result.push(result[result.length - 1] + Math.hypot(dx, dy));
  }
  return result;
}

function formatPicket(distanceM) {
  const total = Math.round(Number(distanceM) * 10) / 10;
  let pk = Math.floor(total / 100);
  let plus = total - pk * 100;

  if (plus >= 99.95) {
    pk += 1;
    plus = 0;
  }

  if (Math.abs(plus - Math.round(plus)) < 1e-9) {
    return `ПК${pk}+${String(Math.round(plus)).padStart(2, "0")}`;
  }

  const plusText = plus.toFixed(1).padStart(4, "0");
  return `ПК${pk}+${plusText}`;
}

function indicesByStation(sM, stepM) {
  const result = new Set();
  for (let station = 0; station <= sM[sM.length - 1] + stepM; station += stepM) {
    let bestIndex = 0;
    let bestDelta = Infinity;
    for (let i = 0; i < sM.length; i += 1) {
      const delta = Math.abs(sM[i] - station);
      if (delta < bestDelta) {
        bestDelta = delta;
        bestIndex = i;
      }
    }
    result.add(bestIndex);
  }
  return Array.from(result).sort((a, b) => a - b);
}

function interpolateValue(target, sM, values) {
  if (target <= sM[0]) {
    return values[0];
  }

  const last = sM.length - 1;
  if (target >= sM[last]) {
    return values[last];
  }

  for (let i = 1; i < sM.length; i += 1) {
    if (sM[i] >= target) {
      const t = (target - sM[i - 1]) / (sM[i] - sM[i - 1]);
      return values[i - 1] + (values[i] - values[i - 1]) * t;
    }
  }

  return values[last];
}

function interpolateByDistance(target, sM, x, y, z) {
  return {
    x: interpolateValue(target, sM, x),
    y: interpolateValue(target, sM, y),
    z: interpolateValue(target, sM, z)
  };
}

function buildTerrain(xPipe, yPipe) {
  const pad = 55;
  const xMin = Math.min(...xPipe) - pad;
  const xMax = Math.max(...xPipe) + pad;
  const yMin = Math.min(...yPipe) - pad;
  const yMax = Math.max(...yPipe) + pad;
  const xg = linspace(xMin, xMax, 90);
  const yg = linspace(yMin, yMax, 70);
  const zg = yg.map(() => zeros(xg.length));
  return { xg, yg, zg };
}

function gradient(values) {
  if (values.length < 2) {
    return zeros(values.length);
  }

  return values.map((_, index) => {
    if (index === 0) {
      return values[1] - values[0];
    }
    if (index === values.length - 1) {
      return values[index] - values[index - 1];
    }
    return (values[index + 1] - values[index - 1]) / 2;
  });
}

function unwrapAngles(angles) {
  if (angles.length === 0) {
    return [];
  }

  const result = [angles[0]];
  let offset = 0;

  for (let i = 1; i < angles.length; i += 1) {
    const delta = angles[i] - angles[i - 1];
    if (delta > Math.PI) {
      offset -= Math.PI * 2;
    } else if (delta < -Math.PI) {
      offset += Math.PI * 2;
    }
    result.push(angles[i] + offset);
  }

  return result;
}

function detectTurnsByCurvature(x, y, threshold = 0.006, minGroupLength = 5) {
  if (x.length < 20) {
    return [];
  }

  const dx = gradient(x);
  const dy = gradient(y);
  const heading = unwrapAngles(dx.map((value, i) => Math.atan2(dy[i], value)));
  const curvature = gradient(heading).map(Math.abs);
  const candidates = curvature
    .map((value, index) => ({ value, index }))
    .filter((item) => item.value > threshold)
    .map((item) => item.index);

  if (candidates.length === 0) {
    return [];
  }

  const groups = [];
  let current = [candidates[0]];

  candidates.slice(1).forEach((candidate) => {
    if (candidate === current[current.length - 1] + 1) {
      current.push(candidate);
    } else {
      if (current.length >= minGroupLength) {
        groups.push(current);
      }
      current = [candidate];
    }
  });

  if (current.length >= minGroupLength) {
    groups.push(current);
  }

  return groups.map((group) => group[Math.floor(group.length / 2)]);
}

function detectObstacleSegments(receivedPoints, zPipe, zGround, sM, threshold = OBSTACLE_CLEARANCE_THRESHOLD_M) {
  const clearance = zPipe.map((z, i) => z - zGround[i]);
  const obstacleMask = receivedPoints.map((point, i) => (
    point.obstacle === true || clearance[i] > threshold
  ));
  const obstacles = [];
  let inObstacle = false;
  let start = 0;

  obstacleMask.forEach((isObstacle, i) => {
    if (isObstacle && !inObstacle) {
      start = i;
      inObstacle = true;
    } else if (!isObstacle && inObstacle) {
      obstacles.push(buildObstacle(start, i - 1, clearance, sM));
      inObstacle = false;
    }
  });

  if (inObstacle) {
    obstacles.push(buildObstacle(start, obstacleMask.length - 1, clearance, sM));
  }

  return obstacles;
}

function buildObstacle(start, end, clearance, sM) {
  let peak = start;
  for (let i = start + 1; i <= end; i += 1) {
    if (clearance[i] > clearance[peak]) {
      peak = i;
    }
  }

  return {
    startIdx: start,
    endIdx: end,
    peakIdx: peak,
    sStart: sM[start],
    sEnd: sM[end],
    sPeak: sM[peak]
  };
}

// Демонстрационная разметка по расстоянию, а не реальные обнаруженные опоры.
function placeSupportsByDistance(sM, stepM = 50) {
  return indicesByStation(sM, stepM);
}

function buildPicketPoints(sM, xPipe, yPipe, zGround, stepM = 100) {
  const distances = [];
  for (let distance = 0; distance <= sM[sM.length - 1] + 0.01; distance += stepM) {
    distances.push(distance);
  }

  return distances.map((distance) => {
    const point = interpolateByDistance(distance, sM, xPipe, yPipe, zGround);
    return {
      x: point.x,
      y: point.y,
      z: point.z + 0.15,
      label: `ПК${Math.floor(distance / 100)}`,
      hover: `Пикет: ${formatPicket(distance)}<br>Расстояние от начала: ${distance.toFixed(1)} м`
    };
  });
}

class RobotSimulator {
  constructor(points) {
    this.allPoints = points.slice();
    this.receivedPoints = [];
  }

  get totalPoints() {
    return this.allPoints.length;
  }

  get receivedCount() {
    return this.receivedPoints.length;
  }

  get finished() {
    return this.receivedCount >= this.totalPoints;
  }

  step(count = 1) {
    const start = this.receivedPoints.length;
    const end = Math.min(start + count, this.totalPoints);
    this.receivedPoints.push(...this.allPoints.slice(start, end));
  }

  finish() {
    this.step(this.totalPoints);
  }

  reset() {
    this.receivedPoints = [];
  }
}

const simulator = new RobotSimulator(ROBOT_POINTS);

function buildEmptyFigure() {
  return {
    data: [],
    layout: {
      title: "Ожидание данных от робота...",
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      margin: { l: 0, r: 0, b: 0, t: 50 },
      scene: {
        xaxis: { title: "X, м" },
        yaxis: { title: "Y, м" },
        zaxis: { title: "Высота, м" },
        aspectratio: { x: 3.2, y: 1.7, z: 0.9 },
        camera: currentCamera()
      }
    },
    config: plotConfig()
  };
}

function buildFigure(receivedPoints) {
  const xPipe = receivedPoints.map((point) => point.x);
  const yPipe = receivedPoints.map((point) => point.y);
  const zPipe = receivedPoints.map((point) => point.z);
  const zGround = receivedPoints.map((point) => point.groundZ ?? 0);

  if (xPipe.length < 2) {
    return buildEmptyFigure();
  }

  const sM = cumulativeDistance(xPipe, yPipe);
  const terrain = buildTerrain(xPipe, yPipe);
  const turnIndices = detectTurnsByCurvature(xPipe, yPipe);
  const obstacleSegments = detectObstacleSegments(receivedPoints, zPipe, zGround, sM);
  const obstacleIndices = obstacleSegments.map((item) => item.peakIdx);
  const supportIndices = placeSupportsByDistance(sM, 50);
  const hasPickets = sM[sM.length - 1] >= 100;
  const pickets = hasPickets ? buildPicketPoints(sM, xPipe, yPipe, zGround) : [];

  const charPoints = [
    { label: "Начало трассы", index: 0, type: "edge" },
    { label: "Конец трассы", index: xPipe.length - 1, type: "edge" },
    ...turnIndices.map((index, i) => ({ label: `Поворот ${i + 1}`, index, type: "turn" })),
    ...obstacleIndices.map((index, i) => ({ label: `Препятствие ${i + 1}`, index, type: "obstacle" }))
  ];

  const projectionHover = sM.map((distance) => (
    `Проекция оси<br>Пикет: ${formatPicket(distance)}<br>Расстояние: ${distance.toFixed(1)} м`
  ));

  const pipeHover = sM.map((distance, i) => (
    `Ось трубопровода<br>Пикет: ${formatPicket(distance)}<br>` +
    `Расстояние: ${distance.toFixed(1)} м<br>` +
    `Просвет: ${(zPipe[i] - zGround[i]).toFixed(2)} м<br>` +
    `Время: ${receivedPoints[i].timestamp}`
  ));

  const supportLineX = [];
  const supportLineY = [];
  const supportLineZ = [];
  supportIndices.forEach((index) => {
    supportLineX.push(xPipe[index], xPipe[index], null);
    supportLineY.push(yPipe[index], yPipe[index], null);
    supportLineZ.push(zGround[index], zPipe[index], null);
  });

  const supportHover = supportIndices.map((index) => (
    `Опора<br>Пикет: ${formatPicket(sM[index])}<br>` +
    `Расстояние: ${sM[index].toFixed(1)} м<br>` +
    `Просвет: ${(zPipe[index] - zGround[index]).toFixed(2)} м`
  ));

  const edgePoints = charPoints.filter((point) => point.type === "edge");
  const turnPoints = charPoints.filter((point) => point.type === "turn");
  const obstaclePoints = charPoints.filter((point) => point.type === "obstacle");
  const totalDistance = sM[sM.length - 1];

  const data = [
    {
      type: "surface",
      x: terrain.xg,
      y: terrain.yg,
      z: terrain.zg,
      colorscale: [[0, "#3a3a3a"], [1, "#3a3a3a"]],
      opacity: 0.84,
      showscale: false,
      name: "Поверхность земли",
      hovertemplate: "Поверхность земли<extra></extra>"
    },
    {
      type: "scatter3d",
      x: xPipe,
      y: yPipe,
      z: zGround.map((z) => z + 0.03),
      mode: "lines",
      line: { color: "rgba(180, 50, 20, 0.55)", width: 4 },
      text: projectionHover,
      hovertemplate: "%{text}<extra></extra>",
      name: "Проекция оси на землю"
    },
    {
      type: "scatter3d",
      x: xPipe,
      y: yPipe,
      z: zPipe,
      mode: "lines",
      line: { color: "red", width: 8 },
      text: pipeHover,
      hovertemplate: "%{text}<extra></extra>",
      name: "Ось трубопровода"
    },
    {
      type: "scatter3d",
      x: [xPipe[xPipe.length - 1]],
      y: [yPipe[yPipe.length - 1]],
      z: [zPipe[zPipe.length - 1] + 0.5],
      mode: "markers",
      marker: { size: 10, color: "lime", symbol: "circle" },
      name: "Позиция робота",
      hovertemplate:
        `Робот<br>X=${xPipe[xPipe.length - 1].toFixed(1)} м<br>` +
        `Y=${yPipe[yPipe.length - 1].toFixed(1)} м<br>` +
        `Время: ${receivedPoints[receivedPoints.length - 1].timestamp}<extra></extra>`
    },
    {
      type: "scatter3d",
      x: supportLineX,
      y: supportLineY,
      z: supportLineZ,
      mode: "lines",
      line: { color: "saddlebrown", width: 5 },
      name: "Опоры",
      hoverinfo: "skip"
    },
    {
      type: "scatter3d",
      x: supportIndices.map((index) => xPipe[index]),
      y: supportIndices.map((index) => yPipe[index]),
      z: supportIndices.map((index) => zPipe[index]),
      mode: "markers",
      marker: { size: 4, color: "saddlebrown" },
      text: supportHover,
      hovertemplate: "%{text}<extra></extra>",
      name: "Точки опор"
    }
  ];

  if (hasPickets) {
    data.push(
      {
        type: "scatter3d",
        x: pickets.map((point) => point.x),
        y: pickets.map((point) => point.y),
        z: pickets.map((point) => point.z),
        mode: "markers",
        marker: { size: 5, color: "black", symbol: "circle" },
        hovertext: pickets.map((point) => point.hover),
        hovertemplate: "%{hovertext}<extra></extra>",
        name: "Пикеты каждые 100 м"
      },
      {
        type: "scatter3d",
        x: pickets.map((point) => point.x),
        y: pickets.map((point) => point.y),
        z: pickets.map((point) => point.z + 0.35),
        mode: "text",
        text: pickets.map((point) => point.label),
        textposition: "top center",
        textfont: { size: 14, color: "indigo" },
        hoverinfo: "skip",
        showlegend: false
      }
    );
  }

  data.push(makeCharacterTrace(edgePoints, xPipe, yPipe, zPipe, zGround, sM, {
    name: "Начало / конец",
    color: "royalblue",
    size: 7,
    symbol: "circle"
  }));

  if (turnPoints.length > 0) {
    data.push(makeCharacterTrace(turnPoints, xPipe, yPipe, zPipe, zGround, sM, {
      name: "Повороты",
      color: "darkorange",
      size: 6,
      symbol: "diamond"
    }));
  }

  if (obstaclePoints.length > 0) {
    data.push(makeCharacterTrace(obstaclePoints, xPipe, yPipe, zPipe, zGround, sM, {
      name: "Препятствия",
      color: "red",
      size: 6,
      symbol: "diamond"
    }));
  }

  return {
    data,
    layout: {
      title: `Трассировочная карта — получено ${xPipe.length} точек | Пройдено: ${totalDistance.toFixed(1)} м (${formatPicket(totalDistance)})`,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      scene: {
        xaxis: { title: "X, м", zerolinecolor: "#d2d9d4", gridcolor: "#e6ece8" },
        yaxis: { title: "Y, м", zerolinecolor: "#d2d9d4", gridcolor: "#e6ece8" },
        zaxis: { title: "Высота, м", zerolinecolor: "#d2d9d4", gridcolor: "#e6ece8" },
        aspectratio: { x: 3.2, y: 1.7, z: 0.9 },
        camera: currentCamera()
      },
      legend: { x: 0.01, y: 0.98, bgcolor: "rgba(255,255,255,0.86)" },
      margin: { l: 0, r: 0, b: 0, t: 50 }
    },
    config: plotConfig()
  };
}

function makeCharacterTrace(points, xPipe, yPipe, zPipe, zGround, sM, style) {
  return {
    type: "scatter3d",
    x: points.map((point) => xPipe[point.index]),
    y: points.map((point) => yPipe[point.index]),
    z: points.map((point) => zPipe[point.index] + 0.9),
    mode: "markers+text",
    marker: { size: style.size, color: style.color, symbol: style.symbol },
    text: points.map((point) => `${point.label}<br>${formatPicket(sM[point.index])}`),
    textposition: "top center",
    textfont: { size: points.length > 2 ? 9 : 11 },
    hovertext: points.map((point) => (
      `${point.label}<br>Пикет: ${formatPicket(sM[point.index])}<br>` +
      `Расстояние: ${sM[point.index].toFixed(1)} м<br>` +
      `Просвет: ${(zPipe[point.index] - zGround[point.index]).toFixed(2)} м`
    )),
    hovertemplate: "%{hovertext}<extra></extra>",
    name: style.name
  };
}

function plotConfig() {
  return {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  };
}

function currentCamera() {
  const radius = 2.62;
  return {
    eye: {
      x: radius * Math.cos(state.cameraAngle),
      y: radius * Math.sin(state.cameraAngle),
      z: 1.08
    }
  };
}

function updateMetrics() {
  const percent = simulator.totalPoints === 0 ? 0 : (simulator.receivedCount / simulator.totalPoints) * 100;
  pointMetric.textContent = `${simulator.receivedCount} / ${simulator.totalPoints} точек`;
  progressFill.style.width = `${percent}%`;

  if (simulator.receivedCount < 2) {
    distanceMetric.textContent = "0.0 м";
    picketMetric.textContent = "ПК0+00";
    return;
  }

  const xPipe = simulator.receivedPoints.map((point) => point.x);
  const yPipe = simulator.receivedPoints.map((point) => point.y);
  const sM = cumulativeDistance(xPipe, yPipe);
  const distance = sM[sM.length - 1];
  distanceMetric.textContent = `${distance.toFixed(1)} м`;
  picketMetric.textContent = formatPicket(distance);
}

function redraw() {
  updateMetrics();

  if (simulator.receivedCount < 2) {
    statusText.textContent = `Ожидание данных от робота... (${simulator.receivedCount} точек)`;
  } else if (simulator.finished) {
    statusText.textContent = "Трасса построена полностью";
  } else if (state.playing) {
    statusText.textContent = "Симуляция поступления координат активна";
  }

  const figure = buildFigure(simulator.receivedPoints);
  Plotly.react(mapElement, figure.data, figure.layout, figure.config);
}

function play() {
  if (simulator.finished || state.playing) {
    return;
  }

  state.playing = true;
  playBtn.textContent = "Идёт...";
  statusText.textContent = "Симуляция поступления координат активна";

  state.playTimer = window.setInterval(() => {
    simulator.step(Number(speedSlider.value));
    redraw();

    if (simulator.finished) {
      pause();
      playBtn.textContent = "Старт";
      statusText.textContent = "Трасса построена полностью";
    }
  }, 350);
}

function pause() {
  state.playing = false;
  window.clearInterval(state.playTimer);
  state.playTimer = null;
  playBtn.textContent = simulator.receivedCount > 0 && !simulator.finished ? "Продолжить" : "Старт";

  if (!simulator.finished) {
    statusText.textContent = `Пауза. Получено: ${simulator.receivedCount} / ${simulator.totalPoints} точек`;
  }
}

function reset() {
  pause();
  simulator.reset();
  playBtn.textContent = "Старт";
  statusText.textContent = "Сброс. Нажмите «Старт» для начала";
  redraw();
}

function finishRoute() {
  pause();
  simulator.finish();
  playBtn.textContent = "Старт";
  statusText.textContent = "Трасса построена полностью";
  redraw();
}

function toggleAutoRotate() {
  state.autoRotate = !state.autoRotate;
  rotateBtn.classList.toggle("active", state.autoRotate);

  if (!state.autoRotate) {
    window.clearInterval(state.rotateTimer);
    state.rotateTimer = null;
    return;
  }

  state.rotateTimer = window.setInterval(() => {
    state.cameraAngle += 0.025;
    Plotly.relayout(mapElement, { "scene.camera": currentCamera() });
  }, 40);
}

playBtn.addEventListener("click", play);
pauseBtn.addEventListener("click", pause);
resetBtn.addEventListener("click", reset);
finishBtn.addEventListener("click", finishRoute);
rotateBtn.addEventListener("click", toggleAutoRotate);

speedSlider.addEventListener("input", () => {
  speedValue.textContent = speedSlider.value;
});

window.addEventListener("resize", () => {
  Plotly.Plots.resize(mapElement);
});

redraw();
