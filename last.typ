== Код программы на языке Wolfram


#text(size: 12pt)[
```wolfram
alpha = 1; h = 0.2; n = 1/h - 1;

fc := Compile[{{x, _Real}}, 
  (x^(3-alpha) - 2(3-alpha)x - 1) / (3-alpha)
]; f[x_?NumericQ] := fc[x];

uexact := Compile[{{x, _Real}}, (x^(3-alpha) - 1) / (3-alpha)];

wc := Compile[{{x, _Real}}, Piecewise[{
  {x,   0<=x<1},
  {2-x, 1<=x<=2}
}, 0]]; w[x_?NumericQ] := wc[x];

phiCompiled = Table[
  With[{j = j}, 
    Compile[{{x, _Real}}, w[x/h - j]]
  ], 
  {j, -1, n - 1}
];
phi = Table[
  With[{i = i},
  phiCompiled[[i]][#] &],
  {i, 1, Length[phiCompiled]}
];

phiD := Table[With[{j = j}, Piecewise[{
    {1/h, h * j <= # < h * (j+1)},
    {-1/h, h * (j+1) <= # <= h * (j+2)}
}, 0]&], {j, -1, n - 1}];

fj = ParallelTable[NIntegrate[
  f[x] * phi[[j+2]][x],
  {x, Max[0,h*j], Min[1,h*(j+2)]},
  WorkingPrecision -> 32, PrecisionGoal -> 8, 
  MaxRecursion -> 20, AccuracyGoal -> 8,
  Method -> {Automatic, "SymbolicProcessing" -> 0}
], {j, -1, n-1}];
t := ParallelTable[
  NIntegrate[
    x^alpha * phiD[[j+2]][x] * phiD[[j+2]][x] + phi[[j+2]][x] * phi[[j+2]][x],
    {x, Max[0,h*j], Min[1,h*(j+2)]},
    WorkingPrecision -> 16, PrecisionGoal -> 8, 
        MaxRecursion -> 20, AccuracyGoal -> 8
  ], {j, -1, n-1}
];



tt := ParallelTable[
  NIntegrate[
    x^alpha * phiD[[j+1]][x] * phiD[[j+2]][x] + phi[[j+1]][x] * phi[[j+2]][x],
    {x, Max[0,h*j], Min[1,h*(j+2)]},
    WorkingPrecision -> 16, PrecisionGoal -> 8, 
        MaxRecursion -> 20, AccuracyGoal -> 8
  ], {j, 0, n-1}
];

k := Length[t]; systemcoef = SparseArray[
  {
      Band[{1, 1}] -> t, 
      Band[{2, 1}] -> tt,
      Band[{1, 2}] -> tt 
  },
  {k, k}
];
a := LinearSolve[systemcoef, fj]; Print[a]
```
]
