#let t1 = figure(table(
  columns: 4,
  table.header([], [$h=0.01$], [$h=0.001$], [$h=0.0001$]),
  [0.0],
  [4.90e-5],
  [6.82e-7],
  [5.09e-3],
  [0.1],
  [1.57e-5],
  [1.57e-7],
  [5.33e-3],
  [0.2],
  [1.11e-5],
  [1.11e-7],
  [4.92e-3],
  [0.3],
  [8.53e-6],
  [8.53e-8],
  [4.36e-3],
  [0.4],
  [6.72e-6],
  [6.72e-8],
  [3.74e-3],
  [0.5],
  [5.29e-6],
  [5.29e-8],
  [3.09e-3],
  [0.6],
  [4.06e-6],
  [4.06e-8],
  [2.43e-3],
  [0.7],
  [2.96e-6],
  [2.96e-8],
  [1.78e-3],
  [0.8],
  [1.94e-6],
  [1.94e-8],
  [1.15e-3],
  [0.9],
  [9.55e-7],
  [9.55e-9],
  [5.55e-4],
), caption: [Абсолютные погрешности, $alpha=1$])

#let t2 = grid(columns: (auto, auto), figure(table(
  columns: 3,
  table.header([], [$h=0.01$], [$h=0.001$]),
  [0.0],
  [2.34e-4],
  [8.26e-6],
  [0.1],
  [1.79e-5],
  [1.76e-7],
  [0.2],
  [1.04e-5],
  [1.03e-7],
  [0.3],
  [7.18e-6],
  [7.12e-8],
  [0.4],
  [5.26e-6],
  [5.22e-8],
  [0.5],
  [3.91e-6],
  [3.88e-8],
  [0.6],
  [2.86e-6],
  [2.84e-8],
  [0.7],
  [2.00e-6],
  [1.98e-8],
  [0.8],
  [1.26e-6],
  [1.25e-8],
  [0.9],
  [5.98e-7],
  [5.94e-9],
), caption: [Абсолютные погрешности, $alpha=1.5$]), figure(table(
  columns: 3,
  table.header([], [$h=0.01$], [$h=0.001$]),
  [0.0],
  [4.07e-4],
  [2.75e-5],
  [0.1],
  [1.13e-5],
  [1.09e-7],
  [0.2],
  [5.85e-6],
  [5.71e-8],
  [0.3],
  [3.80e-6],
  [3.72e-8],
  [0.4],
  [2.66e-6],
  [2.61e-8],
  [0.5],
  [1.91e-6],
  [1.88e-8],
  [0.6],
  [1.36e-6],
  [1.34e-8],
  [0.7],
  [9.25e-7],
  [9.11e-9],
  [0.8],
  [5.68e-7],
  [5.61e-9],
  [0.9],
  [2.65e-7],
  [2.62e-9],
), caption: [Абсолютные погрешности, $alpha=1.8$]))