#import "@preview/cetz:0.3.2"
#import "@preview/cetz-plot:0.1.1"

#import "data.typ"

#let opts = (
  size: (5,5), 
  x-tick-step: 0.25, 
  y-decimals: 10,
  y-label: none,
  y-format: "sci",
  legend: "inner-north"
)

#let style-opts = (
  axes: (stroke: .5pt, tick: (stroke: .5pt)),
  legend: (
    stroke: none, orientation: ttb, item: (spacing: .3), scale: 80%
  )
)

#let p1 = cetz.canvas({
    import cetz.draw: *
    import cetz-plot: *

    set-style(..style-opts)

    let alpha = 1
    
    plot.plot(..opts, y-format: float, y-tick-step: 0.25, {
      plot.add-legend([$h = 0.2$], preview: () => ())

      plot.add(data.d1, style: (stroke: blue))

      plot.add(
        x => (calc.pow(x, (3-alpha)) - 1) / (3 - alpha), 
        domain: (0,1),
        style: (stroke: color.linear-rgb(100%, 0%, 0%, 40%))
      )
    })
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
      }
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
      {
        plot.add(data.d3)
        plot.add-legend([$h = 0.01$], preview: () => ())
      }
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
      {
        plot.add(data.d4)
        plot.add-legend([$h = 0.001$], preview: () => ())
      }
    )
})