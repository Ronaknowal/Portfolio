// Finite, deterministic geometry fixtures. Coordinates are dimensionless unless
// a radius is supplied. Unit-point/tangency tolerances cover binary64 roundoff;
// invalid directions are rejected, not silently turned into different inputs.
const TAU = 2 * Math.PI;
export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export const norm = vector => Math.hypot(...vector);
export const scale = (vector, amount) => vector.map(value => amount * value);
export const add = (a, b) => a.map((value, i) => value + b[i]);
export const subtract = (a, b) => a.map((value, i) => value - b[i]);
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function bounded(value, lower, upper, label) {
  if (!Number.isFinite(value) || value < lower || value > upper) {
    throw new RangeError(`${label} must be finite and between ${lower} and ${upper}.`);
  }
  return value;
}
function vector3(vector, label) {
  if (!Array.isArray(vector) || vector.length !== 3 || vector.some(value => !Number.isFinite(value) || Math.abs(value) > 10)) {
    throw new RangeError(`${label} needs three finite components with magnitude at most 10.`);
  }
}
function unitPoint(point) {
  vector3(point, 'Point');
  if (Math.abs(norm(point) - 1) > 1e-10) throw new RangeError('Use a unit point.');
}
function tangentAt(point, vector) {
  unitPoint(point);
  vector3(vector, 'Tangent');
  if (Math.abs(dot(point, vector)) > 1e-10 * Math.max(1, norm(vector))) {
    throw new RangeError('The supplied vector must be tangent at the point.');
  }
}
function frozen(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(frozen);
    Object.freeze(value);
  }
  return value;
}
export function formatGeometry(value, digits = 5) {
  if (value === null) return 'unavailable';
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) > 1e5) return value.toExponential(2);
  return String(Number(value.toFixed(digits)));
}
export const radians = degrees => degrees * Math.PI / 180;
export function circleCharts(degrees = 0) {
  bounded(degrees, 0, 360, 'Circle angle');
  const angle = radians(degrees);
  const alpha = degrees === 180 ? null : degrees > 180 ? angle - TAU : angle;
  const beta = degrees === 0 || degrees === 360 ? null : angle;
  return frozen({
    degrees,
    angle,
    point: [Math.cos(angle), Math.sin(angle)],
    tangent: [-Math.sin(angle), Math.cos(angle)],
    alpha: degrees === 360 ? 0 : alpha,
    beta,
    transition: alpha === null || beta === null ? null : beta - (degrees === 360 ? 0 : alpha)
  });
}
export function spherePatch(thetaDegrees = 60, phiDegrees = 35, radius = 1) {
  bounded(thetaDegrees, 15, 165, 'Polar angle');
  bounded(phiDegrees, -180, 180, 'Longitude');
  bounded(radius, 0.5, 3, 'Radius');
  const theta = radians(thetaDegrees),
    phi = radians(phiDegrees);
  const point = scale([Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)], radius);
  const thetaBasis = scale([Math.cos(theta) * Math.cos(phi), Math.cos(theta) * Math.sin(phi), -Math.sin(theta)], radius);
  const phiBasis = scale([-Math.sin(theta) * Math.sin(phi), Math.sin(theta) * Math.cos(phi), 0], radius);
  return frozen({
    point,
    thetaBasis,
    phiBasis,
    theta,
    phi,
    radius,
    metric: [[radius ** 2, 0], [0, (radius * Math.sin(theta)) ** 2]],
    areaWeight: radius ** 2 * Math.sin(theta)
  });
}
export function metricDifferential(shear = 1, verticalCost = 1, directionDegrees = 45) {
  bounded(shear, -1.5, 1.5, 'Shear');
  bounded(verticalCost, 0.5, 3, 'Vertical cost');
  bounded(directionDegrees, 0, 360, 'Direction angle');
  const costSquared = verticalCost ** 2;
  const metric = [[1, shear], [shear, shear ** 2 + costSquared]];
  const covector = [2, 2 * shear - 1];
  const gradient = [2 + shear / costSquared, -1 / costSquared];
  const worldGradient = [2, -1 / costSquared];
  const transform = ([u, v]) => [u + shear * v, v];
  const worldDirection = [Math.cos(radians(directionDegrees)), Math.sin(radians(directionDegrees)) / verticalCost];
  const coordinateDirection = [worldDirection[0] - shear * worldDirection[1], worldDirection[1]];
  const worldEllipse = Array.from({
    length: 129
  }, (_, index) => [Math.cos(TAU * index / 128), Math.sin(TAU * index / 128) / verticalCost]);
  return frozen({
    shear,
    verticalCost,
    metric,
    covector,
    gradient,
    worldGradient,
    coordinateDirection,
    worldDirection,
    directionalDerivative: dot(covector, coordinateDirection),
    gradientNorm: Math.sqrt(4 + 1 / costSquared),
    coordinateEllipse: worldEllipse.map(([x, y]) => [x - shear * y, y]),
    worldEllipse,
    reconstructedGradient: transform(gradient)
  });
}
export function sphereProjection(point, ambient) {
  unitPoint(point);
  vector3(ambient, 'Ambient vector');
  return frozen(subtract(ambient, scale(point, dot(point, ambient))));
}
function sinc(value) {
  if (Math.abs(value) < 1e-4) {
    const square = value * value;
    return 1 - square / 6 + square * square / 120 - square * square * square / 5040;
  }
  return Math.sin(value) / value;
}
export function sphereExp(point, tangent) {
  tangentAt(point, tangent);
  const length = norm(tangent);
  bounded(length, 0, 8, 'Tangent length');
  return frozen(add(scale(point, Math.cos(length)), scale(tangent, sinc(length))));
}
export function sphereRetract(point, tangent) {
  tangentAt(point, tangent);
  const candidate = add(point, tangent);
  return frozen(scale(candidate, 1 / norm(candidate)));
}
export function sphereAngle(first, second) {
  unitPoint(first);
  unitPoint(second);
  return Math.atan2(norm(cross(first, second)), dot(first, second));
}
export function sphereLog(first, second) {
  unitPoint(first);
  unitPoint(second);
  const cosine = dot(first, second);
  // The double cross product avoids cancelling nearly identical coordinates.
  const tangent = cross(cross(first, second), first);
  const sine = norm(cross(first, second));
  if (cosine < 0 && sine < 1e-8) throw new RangeError('The shortest logarithm is nonunique or numerically unresolved near the antipode.');
  if (sine === 0) return frozen([0, 0, 0]);
  return frozen(scale(tangent, Math.atan2(sine, cosine) / sine));
}
export function sphereTransport(first, second, tangent) {
  tangentAt(first, tangent);
  unitPoint(second);
  const denominator = 1 + dot(first, second);
  if (denominator < 1e-6) throw new RangeError('Choose a specified route away from antipodal endpoints.');
  return frozen(subtract(tangent, scale(add(first, second), dot(tangent, second) / denominator)));
}
export function sphereArc(angleDegrees = 90, fraction = 0.5, radius = 1) {
  bounded(angleDegrees, 0, 180, 'Separation angle');
  bounded(fraction, 0, 1, 'Path fraction');
  bounded(radius, 0.5, 3, 'Radius');
  const angle = radians(angleDegrees);
  const point = theta => [radius * Math.cos(theta), radius * Math.sin(theta)];
  return frozen({
    angleDegrees,
    fraction,
    radius,
    angle,
    first: point(0),
    second: point(angle),
    shortPoint: point(fraction * angle),
    longPoint: point(-fraction * (TAU - angle)),
    chordPoint: [(1 - fraction) * radius + fraction * radius * Math.cos(angle), fraction * radius * Math.sin(angle)],
    shortLength: radius * angle,
    longLength: radius * (TAU - angle),
    chordLength: 2 * radius * Math.sin(angle / 2),
    uniqueShortest: angleDegrees !== 180,
    logarithmStatus: angleDegrees === 180 ? 'nonunique at antipodes' : angleDegrees === 0 ? 'zero tangent' : 'unique shorter tangent'
  });
}
export function sphereStep(startDegrees = 0, learningRate = 0.1) {
  bounded(startDegrees, -180, 180, 'Starting direction');
  bounded(learningRate, 0, 1.5, 'Step size');
  const angle = radians(startDegrees);
  const point = [Math.cos(angle), Math.sin(angle), 0];
  const ambientGradient = [1, 2, 0];
  const tangentGradient = sphereProjection(point, ambientGradient);
  const tangentStep = scale(tangentGradient, -learningRate);
  const candidate = add(point, tangentStep);
  const rawAmbientStep = add(point, scale(ambientGradient, -learningRate));
  const retracted = sphereRetract(point, tangentStep);
  const exponential = sphereExp(point, tangentStep);
  const cost = position => dot(ambientGradient, position);
  return frozen({
    startDegrees,
    learningRate,
    point,
    ambientGradient,
    tangentGradient,
    tangentStep,
    candidate,
    rawAmbientStep,
    retracted,
    exponential,
    tangentLength: norm(tangentStep),
    retractionAngle: Math.atan(norm(tangentStep)),
    candidateNorm: norm(candidate),
    rawAmbientNorm: norm(rawAmbientStep),
    costs: {
      original: cost(point),
      retracted: cost(retracted),
      exponential: cost(exponential)
    }
  });
}
export function sphereBands(radius = 1) {
  bounded(radius, 0.5, 3, 'Radius');
  const bands = [[0, 30], [30, 60], [60, 90]].map(([lower, upper]) => ({
    lower,
    upper,
    area: 2 * Math.PI * radius ** 2 * (Math.cos(radians(lower)) - Math.cos(radians(upper))),
    surfaceFraction: (Math.cos(radians(lower)) - Math.cos(radians(upper))) / 2,
    equalAngleFraction: (upper - lower) / 180
  }));
  return frozen({
    radius,
    bands,
    fullArea: 4 * Math.PI * radius ** 2
  });
}
export function polarConnection(time = 1, height = 1) {
  bounded(time, -2, 2, 'Line position');
  bounded(height, 0.25, 2, 'Line height');
  const r = Math.hypot(time, height),
    theta = Math.atan2(height, time);
  const radialVelocity = time / r,
    angularVelocity = -height / r ** 2;
  const radialAcceleration = height ** 2 / r ** 3,
    angularAcceleration = 2 * height * time / r ** 4;
  const radialCorrection = -r * angularVelocity ** 2;
  const angularCorrection = 2 * radialVelocity * angularVelocity / r;
  return frozen({
    time,
    height,
    point: [time, height],
    r,
    theta,
    radialBasis: [time / r, height / r],
    angularBasis: [-height, time],
    radialVelocity,
    angularVelocity,
    radialAcceleration,
    angularAcceleration,
    radialCorrection,
    angularCorrection,
    covariantAcceleration: [radialAcceleration + radialCorrection, angularAcceleration + angularCorrection],
    metric: [[1, 0], [0, r * r]],
    christoffel: {
      radialAngularAngular: -r,
      angularRadialAngular: 1 / r
    },
    cartesianVelocity: [radialVelocity * time / r - angularVelocity * height, radialVelocity * height / r + angularVelocity * time]
  });
}
export function transportTriangle(wedgeDegrees = 90, initialDegrees = 0, progress = 3, reverse = false, radius = 1) {
  bounded(wedgeDegrees, 15, 150, 'Longitude wedge');
  bounded(initialDegrees, -180, 180, 'Initial arrow angle');
  bounded(progress, 0, 3, 'Route progress');
  bounded(radius, 0.5, 3, 'Radius');
  if (typeof reverse !== 'boolean') throw new RangeError('Route reversal must be boolean.');
  const wedge = radians(wedgeDegrees);
  const north = [0, 0, 1],
    first = [1, 0, 0],
    second = [Math.cos(wedge), Math.sin(wedge), 0];
  const vertices = reverse ? [north, second, first, north] : [north, first, second, north];
  const initial = [Math.cos(radians(initialDegrees)), Math.sin(radians(initialDegrees)), 0];
  const vectors = [initial];
  for (let leg = 0; leg < 3; leg++) vectors.push(sphereTransport(vertices[leg], vertices[leg + 1], vectors[leg]));
  const leg = Math.min(2, Math.floor(progress));
  const fraction = progress === 3 ? 1 : progress - leg;
  const velocity = sphereLog(vertices[leg], vertices[leg + 1]);
  const currentUnit = sphereExp(vertices[leg], scale(velocity, fraction));
  const currentVector = sphereTransport(vertices[leg], currentUnit, vectors[leg]);
  const paths = vertices.slice(0, 3).map((point, index) => {
    const tangent = sphereLog(point, vertices[index + 1]);
    return Array.from({
      length: 33
    }, (_, i) => scale(sphereExp(point, scale(tangent, i / 32)), radius));
  });
  const finalTurn = Math.atan2(dot(north, cross(initial, vectors[3])), dot(initial, vectors[3]));
  return frozen({
    wedgeDegrees,
    initialDegrees,
    progress,
    reverse,
    radius,
    paths,
    vertices: vertices.map(point => scale(point, radius)),
    vectors,
    labels: reverse ? ['N', 'B', 'A', 'N'] : ['N', 'A', 'B', 'N'],
    point: scale(currentUnit, radius),
    currentVector,
    initial,
    finalVector: vectors[3],
    currentNorm: norm(currentVector),
    tangencyResidual: dot(currentUnit, currentVector),
    finalTurn,
    expectedTurn: (reverse ? -1 : 1) * wedge,
    enclosedArea: radius ** 2 * wedge,
    curvature: 1 / radius ** 2
  });
}
export function warpedMetric(kind = 'sphere', coordinate = 1, radius = 1) {
  bounded(radius, 0.5, 3, 'Radius');
  bounded(coordinate / radius, 0.1, 2, 'Scaled coordinate');
  let a, first, second;
  if (kind === 'plane') [a, first, second] = [coordinate, 1, 0];else if (kind === 'cylinder') [a, first, second] = [radius, 0, 0];else if (kind === 'sphere') [a, first, second] = [radius * Math.sin(coordinate / radius), Math.cos(coordinate / radius), -Math.sin(coordinate / radius) / radius];else if (kind === 'hyperbolic') [a, first, second] = [radius * Math.exp(coordinate / radius), Math.exp(coordinate / radius), Math.exp(coordinate / radius) / radius];else throw new RangeError('Choose plane, cylinder, sphere or hyperbolic.');
  return frozen({
    kind,
    coordinate,
    radius,
    a,
    first,
    second,
    metric: [[1, 0], [0, a * a]],
    gammaUVV: -a * first,
    gammaVUV: first / a,
    riemannUVVU: -a * second,
    curvature: -second / a
  });
}
export function curvatureComparison(radius = 1, scaledDistance = 0.75) {
  bounded(radius, 0.5, 3, 'Radius');
  bounded(scaledDistance, 0.05, 1.5, 'Distance in radius units');
  const distance = radius * scaledDistance;
  const kinds = ['plane', 'cylinder', 'sphere', 'hyperbolic'];
  const rows = kinds.map(kind => {
    const metric = warpedMetric(kind, radius * .75, radius);
    const coefficient = kind === 'sphere' ? radius * Math.sin(scaledDistance) : kind === 'hyperbolic' ? radius * Math.sinh(scaledDistance) : distance;
    const area = kind === 'sphere' ? 4 * Math.PI * radius ** 2 * Math.sin(scaledDistance / 2) ** 2 : kind === 'hyperbolic' ? 4 * Math.PI * radius ** 2 * Math.sinh(scaledDistance / 2) ** 2 : Math.PI * distance ** 2;
    return {
      ...metric,
      jacobiCoefficient: coefficient,
      circumference: TAU * coefficient,
      area
    };
  });
  return frozen({
    radius,
    scaledDistance,
    distance,
    rows
  });
}
export function covariancePaths(ratio = 4, fraction = 0.5) {
  bounded(ratio, 2, 9, 'Variance ratio');
  bounded(fraction, 0, 1, 'Path fraction');
  const logarithm = Math.log(ratio);
  return frozen({
    ratio,
    fraction,
    first: [1, ratio],
    second: [ratio, 1],
    arithmetic: [1 + fraction * (ratio - 1), ratio - fraction * (ratio - 1)],
    affine: [Math.exp(fraction * logarithm), Math.exp((1 - fraction) * logarithm)],
    distance: Math.SQRT2 * logarithm
  });
}
