#import "symbols.typ": *

#let doc(body, bibliography-file: "references.bib") = {
  set text(size: 14pt, lang: "ru")
  set heading(numbering: "1.1.1")

  body
  
  pagebreak()
  
  heading(level: 1, "Список литературы")
  bibliography(bibliography-file, full: true, title: none)
}

#show: doc.with(bibliography-file: "references.bib")

= Дифференциальные уравнения с сильным вырождением
ДУ с сильным вырождением называют уравнения, в которых старшая производная умножается на функцию, которая может обращаться в ноль в области определения. Например, для второго порядка общий вид будет:
$
  a(x,t)u_t = b(x,t)u_(x x) + c(x,t)u_x + d(x)u + f(x,t), quad x,t in Omega times T,\
  exists x_0,t_0 in Omega times T: a(x_0,t_0) = 0
$
Или в стационарном случае
$
  a(x)u_(x x) + b(x)u_x + c(x) u = f(x), quad x in Omega,\
  exists x_0 in Omega: a(x_0) = 0
$
Такие уравнения требуют особых численных методов решения, также возникают естественные граничные условия.

Они могут возникать в задачах теплопроводности, гидродинамики.

Для приложений требуется задавать некоторые граничные условия, которые задают начальные параметры системы. Случай, когда условие состоит в задании начальных значений искомой функции на границе, называется *первой краевой задачей*, или *задачей Дирихле*.

= Аппроксимация решения первой краевой задачи для вырождающихся одномерных дифференциальных уравнений второго порядка

== Вид уравнения
В данной главе работы будет рассмотрено уравнение следующего вида:
$
  -d/dx [x^alpha p(x) du/dx] + q(x) u = f(x), quad 0 < x < 1, quad f in L2(0,1),\
  q "измерима, ограничена, неотрицательна на" [0,1]\
  alpha = const > 0, space p in C^1 [0,1], space p(x) >= p_0 = const > 0\
  u in WW_2^1(0,1)
$
== Аппроксимация в пространстве $WW_2^1$
=== Основа вариацонно-сеточного метода
Пусть в гильбертовом пространстве $H$ действует линейный положительно-определенный оператор $A$ и требуется найти решение уравнения
$
  A u = f, quad f in H
$
Принято вводить функционал энергии и энергетическую норму с энергетическим произведением:
$
  [u,v]_A := (A u,v),\
  norm(u)_A^2 := [u,u]_A = (A u,u),\
  cal(F)(u) := 1/2[u,u]_A - (f,u)
$
При положительной определенности оператора $A$ функционал энергии является  выпуклым, из чего следует, что ноль его производной является точкой минимума.
$
  cal(F)'(u)h = [u,h]_A - (f,h)\
  cal(F)'(u_0) = 0 <=> forall h in H quad [u_0,h]_A = (f,h) <=> A u_0 = f\
$
Если решение $u_0$ аппроксимируется в конечномерном пространстве $H_n$ с энергетическим произведением $[dot,dot]_A$, то критерием минимальности функционала энергии в точке $u_n in H_n$ будет следующая система линейных уравнений:
$
  u_n = sum_(k=1)^n a_k phi_k, " где Lin" {phi_1,dots,phi_n} = H_n,\
  cal(F)'(u_n) = 0 <=> partial/(partial a_k) cal(F)(u_n) = 0, space k = 1,...,n\
  partial/(partial a_k) cal(F)(u_n) = partial/(partial a_k) [1/2 sum_(i=1)^n sum_(k=1)^n a_i a_k [phi_i, phi_k]_A - sum_(i=1)^n a_i (f, phi_i)] =\
  = sum_(i=1)^n a_i [phi_i, phi_k]_A - (f, phi_k) =0, space k = 1,...,n\
$
Приведенная схема называется методом Ритца.