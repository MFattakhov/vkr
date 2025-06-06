#import "@preview/cetz:0.3.2"
#import "@preview/cetz-plot:0.1.1"

#import "@preview/lilaq:0.3.0" as lq

#import "data.typ"

#let opts = (size: (4.3, 4.3), x-tick-step: 0.25, y-decimals: 10, y-label: none, y-format: "sci", legend: "inner-north")

#let style-opts = (
  axes: (stroke: .5pt, tick: (stroke: .5pt)),
  legend: (stroke: none, orientation: ttb, item: (spacing: .3), scale: 80%),
)

#let p1 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-format: float,
    y-tick-step: 0.25,
    {
      plot.add-legend([$h = 0.2$], preview: () => ())

      plot.add(data.d1, style: (stroke: blue))

      plot.add(
        x => (calc.pow(x, (3 - alpha)) - 1) / (3 - alpha),
        domain: (0, 1),
        style: (stroke: color.linear-rgb(100%, 0%, 0%, 40%)),
      )
    },
  )
})

#let p2 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-tick-step: 0.002,
    {
      plot.add(data.d2)
      plot.add-legend([$h = 0.2$], preview: () => ())
    },
  )
})

#let p3 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-tick-step: 0.00002,
    x-label: [],
    {
      plot.add(data.d3)
      plot.add-legend([$h = 0.01$], preview: () => ())
    },
  )
})

#let p4 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-tick-step: 0.0000002,
    x-label: [],
    {
      plot.add(data.d4)
      plot.add-legend([$h = 0.001$], preview: () => ())
    },
  )
})

#let p5 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let w0_first = x => -2 * calc.pow(x, 3) + 3 * calc.pow(x, 2)
  let w0_second = x => 2 * calc.pow(x, 3) - 9 * calc.pow(x, 2) + 12 * x - 4

  plot.plot(
    size: (5, 5),
    x-tick-step: 0.5,
    y-tick-step: 0.25,
    y-min: -0.25 + 0.001,
    y-max: 1.25 - 0.001,
    legend: "inner-north",
    x-label: [],
    {
      plot.add-legend([$omega_0$], preview: () => ())

      let domain = (-0.25, 2.25)
      let style = (style: (stroke: red))

      let domain_first_outer = (-0.25, 0)
      let domain_first = (0, 1)
      let domain_second = (1, 2)
      let domain_second_outer = (2, 2.25)

      plot.add(w0_first, domain: domain_first, samples: 100, ..style)
      plot.add(w0_second, domain: domain_second, samples: 100, ..style)
      plot.add(x => 0, domain: domain_first_outer, samples: 10, ..style)
      plot.add(x => 0, domain: domain_second_outer, samples: 10, ..style)
    },
  )
})

#let p6 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let w1_first = x => calc.pow(x, 3) - calc.pow(x, 2)
  let w1_second = x => calc.pow(x, 3) - 5 * calc.pow(x, 2) + 8 * x - 4

  plot.plot(
    size: (5, 5),
    x-tick-step: 0.5,
    y-tick-step: 0.0725,
    y-min: -0.25 + 0.001,
    y-max: 0.25 - 0.001,
    legend: "inner-north",
    y-label: [],
    x-label: [],
    {
      plot.add-legend([$omega_1$], preview: () => ())

      let domain = (-0.25, 2.25)
      let style = (style: (stroke: red))

      let domain_first_outer = (-0.25, 0)
      let domain_first = (0, 1)
      let domain_second = (1, 2)
      let domain_second_outer = (2, 2.25)

      plot.add(w1_first, domain: domain_first, samples: 100, ..style)
      plot.add(w1_second, domain: domain_second, samples: 100, ..style)
      plot.add(x => 0, domain: domain_first_outer, samples: 10, ..style)
      plot.add(x => 0, domain: domain_second_outer, samples: 10, ..style)
    },
  )
})

#let p7 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  plot.plot(
    ..opts,
    y-format: float,
    y-tick-step: 0.25,
    {
      plot.add-legend([$h = 1 slash 4$], preview: () => ())

      plot.add(data.d5, style: (stroke: blue))

      plot.add(
        x => calc.pow(x + 1, 3 - 1.5) * calc.pow(1 - x, 2),
        domain: (0, 1),
        style: (stroke: color.linear-rgb(100%, 0%, 0%, 40%)),
      )
    },
  )
})

#let p8 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-tick-step: 0.01,
    {
      plot.add(data.d6)
      plot.add-legend([$h = 1 slash 4$], preview: () => ())
    },
  )
})

#let p9 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  plot.plot(
    ..opts,
    y-format: float,
    y-tick-step: 0.5,
    {
      plot.add-legend([$h = 1 slash 4$], preview: () => ())

      plot.add(data.d7, style: (stroke: blue))

      plot.add(
        x => 1.5 * calc.pow(1 - x, 2) * calc.pow(x + 1, 0.5) + calc.pow(x + 1, 1.5) * (2 * x - 2),
        domain: (0, 1),
        style: (stroke: color.linear-rgb(100%, 0%, 0%, 40%)),
      )
    },
  )
})

#let p10 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-tick-step: 0.1,
    {
      plot.add(data.d8)
      plot.add-legend([$h = 1 slash 4$], preview: () => ())
    },
  )
})

#let p11 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1.8

  plot.plot(
    ..opts,
    y-format: float,
    y-tick-step: 0.025,
    {
      plot.add-legend([$h = 1 slash 8$], preview: () => ())

      plot.add(data.d9, style: (stroke: blue))

      plot.add(
        x => calc.pow(x, 3 - alpha) * calc.pow(1 - x, 2),
        domain: (0, 1),
        style: (stroke: color.linear-rgb(100%, 0%, 0%, 40%)),
      )
    },
  )
})

#let p12 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-tick-step: 0.001,
    {
      plot.add(data.d10)
      plot.add-legend([$h = 1 slash 8$], preview: () => ())
    },
  )
})

#let p13 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1.8

  plot.plot(
    ..opts,
    y-format: float,
    y-tick-step: 0.5,
    {
      plot.add-legend([$h = 1 slash 8$], preview: () => ())

      plot.add(data.d11, style: (stroke: blue))

      plot.add(
        x => (3 - alpha) * calc.pow(1 - x, 2) * calc.pow(x, 2 - alpha) - 2 * (1 - x) * calc.pow(x, 3 - alpha),
        domain: (0, 1),
        style: (stroke: color.linear-rgb(100%, 0%, 0%, 40%)),
      )
    },
  )
})

#let p14 = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  let alpha = 1

  plot.plot(
    ..opts,
    y-tick-step: 0.1,
    {
      plot.add(data.d12)
      plot.add-legend([$h = 1 slash 8$], preview: () => ())
    },
  )
})

#let p_comparison = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(..style-opts)

  plot.plot(
    ..opts,
    y-tick-step: 0.25,
    y-format: float,
    {
      plot.add(data.exact, style: (stroke: blue), samples: 2)
      plot.add(data.method, style: (stroke: orange), samples: 2)
      plot.add(data.linear, style: (stroke: green), samples: 2)
      plot.add(data.hermite, style: (stroke: red, dash: "dashed"), samples: 2)
      plot.add(data.cubic_spline, style: (stroke: purple, dash: "dotted"), samples: 2)
      plot.add(
        data.nodes,
        style: (stroke: none),
        mark: "o",
        mark-style: (fill: black, stroke: none),
        mark-size: 0.1,
      )
      plot.add-legend([$h = 1 slash 4$], preview: () => ())
    },
  )
})

#let p_comparison_errs = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(
    axes: (stroke: .5pt, tick: (stroke: .5pt)),
    legend: (stroke: 0.25pt, orientation: ttb, item: (spacing: .3), scale: 50%),
  )

  plot.plot(
    x-tick-step: 0.25,
    y-decimals: 10,
    y-label: none,
    x-label: none,
    y-format: "sci",
    y-tick-step: 0.01,
    legend: "inner-north",
    size: (13, 5),
    {
      plot.add(data.method_abs_err, style: (stroke: rgb("#ff7f0e")), samples: 2, label: text(size:10pt)[Предлагаемый метод])
      plot.add(data.linear_abs_err, style: (stroke: rgb("#2ca02c")), samples: 2, label: text(size:10pt)[Линейная интерполяция])
      plot.add(
        data.hermite_abs_err,
        style: (stroke: rgb("#d62728"), dash: "dashed"),
        samples: 2,
        label: text(size:10pt)[Эрмитова $C^1$ интерполяция],
      )
      plot.add(
        data.cubic_spline_abs_err,
        style: (stroke: rgb("#9467bd"), dash: "dotted"),
        samples: 2,
        label: text(size:10pt)[Кубическая $C^2$ интерполяция],
      )
    },
  )
})

#let p_prime_comparison_errs = cetz.canvas({
  import cetz.draw: *
  import cetz-plot: *

  set-style(
    axes: (stroke: .5pt, tick: (stroke: .5pt)),
    legend: (stroke: 0.25pt, orientation: ttb, item: (spacing: .3), scale: 50%),
  )

  plot.plot(
    x-tick-step: 0.25,
    y-decimals: 10,
    y-label: none,
    x-label: none,
    y-format: "sci",
    y-tick-step: 0.1,
    legend: "inner-north",
    size: (13, 5),
    {
      plot.add(data.method_prime_abs_err, style: (stroke: rgb("#ff7f0e")), samples: 2, label: text(size:10pt)[Предлагаемый метод])
      plot.add(data.linear_prime_abs_err, style: (stroke: rgb("#2ca02c")), samples: 2, label: text(size:10pt)[Линейная интерполяция])
      plot.add(
        data.hermite_prime_abs_err,
        style: (stroke: rgb("#d62728"), dash: "dashed"),
        samples: 2,
        label: text(size:10pt)[Эрмитова $C^1$ интерполяция],
      )
      plot.add(
        data.cubic_spline_prime_abs_err,
        style: (stroke: rgb("#9467bd"), dash: "dotted"),
        samples: 2,
        label: text(size:10pt)[Кубическая $C^2$ интерполяция],
      )
    },
  )
})
