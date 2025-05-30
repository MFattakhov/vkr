

#let t1 = figure(
  table(
    columns: 4,
    table.header([], [$h=0.01$], [$h=0.001$], [$h=0.0001$]),
    [0.0], [4.90e-5], [6.82e-7], [8.42e-09],
    [0.1], [1.57e-5], [1.57e-7], [1.21e-09],
    [0.2], [1.11e-5], [1.11e-7], [6.73e-10],
    [0.3], [8.53e-6], [8.53e-8], [3.40e-10],
    [0.4], [6.72e-6], [6.72e-8], [1.15e-10],
    [0.5], [5.29e-6], [5.29e-8], [2.43e-11],
    [0.6], [4.06e-6], [4.06e-8], [7.90e-11],
    [0.7], [2.96e-6], [2.96e-8], [8.42e-11],
    [0.8], [1.94e-6], [1.94e-8], [6.29e-11],
    [0.9], [9.55e-7], [9.55e-9], [3.08e-11],
  ),
  caption: [Абсолютные погрешности, $alpha=1$],
  kind: table,
)

#let t2 = grid(
  columns: (1fr, 1fr),
  [#figure(
    table(
      columns: 3,
      table.header([], [$h=0.01$], [$h=0.001$]),
      [0.0], [2.34e-4], [8.26e-6],
      [0.1], [1.79e-5], [1.76e-7],
      [0.2], [1.04e-5], [1.03e-7],
      [0.3], [7.18e-6], [7.12e-8],
      [0.4], [5.26e-6], [5.22e-8],
      [0.5], [3.91e-6], [3.88e-8],
      [0.6], [2.86e-6], [2.84e-8],
      [0.7], [2.00e-6], [1.98e-8],
      [0.8], [1.26e-6], [1.25e-8],
      [0.9], [5.98e-7], [5.94e-9],
    ),
    caption: [Абсолютные погрешности, \ $alpha=1.5$],
  )<t2>],
  [#figure(
    table(
      columns: 3,
      table.header([], [$h=0.01$], [$h=0.001$]),
      [0.0], [4.07e-4], [2.75e-5],
      [0.1], [1.13e-5], [1.09e-7],
      [0.2], [5.85e-6], [5.71e-8],
      [0.3], [3.80e-6], [3.72e-8],
      [0.4], [2.66e-6], [2.61e-8],
      [0.5], [1.91e-6], [1.88e-8],
      [0.6], [1.36e-6], [1.34e-8],
      [0.7], [9.25e-7], [9.11e-9],
      [0.8], [5.68e-7], [5.61e-9],
      [0.9], [2.65e-7], [2.62e-9],
    ),
    caption: [Абсолютные погрешности, \ $alpha=1.8$],
  )<t3>],
)

#let t3 = figure(
  table(
    columns: 3,
    table.header([], [$h=0.01$], [$h=0.002$]),
    [0.0], [1.18e-4], [5.99e-6],
    [0.1], [3.57e-5], [1.43e-6],
    [0.2], [2.30e-5], [9.21e-7],
    [0.3], [1.54e-5], [6.17e-7],
    [0.4], [9.98e-6], [3.99e-7],
    [0.5], [5.88e-6], [2.35e-7],
    [0.6], [2.82e-6], [1.13e-7],
    [0.7], [7.03e-7], [2.81e-8],
    [0.8], [4.97e-7], [1.99e-8],
    [0.9], [7.50e-7], [3.00e-8],
  ),
  caption: [Абсолютные погрешности, \ $alpha=1$],
)

#let t4 = figure(
  table(
    columns: 3,
    table.header([], [$h=0.01$], [$h=0.002$]),
    [0.0], [1.35e-4], [4.65e-5],
    [0.1], [2.27e-5], [9.63e-7],
    [0.2], [1.09e-5], [4.62e-7],
    [0.3], [4.05e-6], [1.78e-7],
    [0.4], [3.01e-7], [1.78e-9],
    [0.5], [2.98e-6], [1.12e-7],
    [0.6], [4.38e-6], [1.70e-7],
    [0.7], [4.67e-6], [1.84e-7],
    [0.8], [4.00e-6], [1.58e-7],
    [0.9], [2.42e-6], [9.60e-8],
  ),
  caption: [Абсолютные погрешности, \ $alpha=1.5$],
)

#let t5 = figure(
  table(
    columns: 4,
    table.header([], [$h=0.1$], [$h=0.01$], [$h=0.001$]),
    [0.1], [1.91e-3], [1.99e-5], [1.87e-7],
    [0.2], [1.45e-3], [1.50e-5], [1.45e-7],
    [0.3], [9.92e-4], [1.02e-5], [9.95e-8],
    [0.4], [6.17e-4], [6.35e-6], [6.20e-8],
    [0.5], [3.31e-4], [3.44e-6], [3.34e-8],
    [0.6], [1.28e-4], [1.37e-6], [1.30e-8],
    [0.7], [7.73e-8], [5.89e-8], [1.65e-10],
    [0.8], [6.01e-5], [5.67e-7], [5.91e-9],
    [0.9], [5.85e-5], [5.69e-7], [5.80e-9],
  ),
  caption: [Абсолютные погрешности, \ $alpha=1.8$],
)
