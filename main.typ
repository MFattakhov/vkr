#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot, chart

#import "symbols.typ": *

#import "tables.typ"
#import "plots.typ"

#let doc(body, bibliography-file: "references.bib") = {
  set text(size: 12pt, lang: "ru")
  set heading(numbering: "1.1.1")
  set math.equation(numbering: "(1)")
  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      // Override equation references.
      link(el.location(), numbering(el.numbering, ..counter(eq).at(el.location())))
    } else {
      // Other references as usual.
      it
    }
  }

  set page(footer: {
    align(center, "Санкт-Петербург \n 2025")
  })
  align(center, text("Санкт-Петербургский государственный университет"))
  v(0.6fr)
  v(.0fr)

  let authors = ("Фаттахов Марат Русланович",)
  pad(top: 0.7em, grid(columns: (1fr,) * calc.min(3, authors.len()), gutter: 1em, ..authors.map(author =>
  align(center, text(size: 16pt, emph(strong(author)))))))
  v(.1fr)
  align(center, "Выпускная квалификационная работа")
  let title = "Решение дифференциальных уравнений с сильным вырождением"
  align(center, text(1.75em, weight: 500, title))
  v(.1fr)
  align(center, "Уровень образования: бакалавриат")

  v(.2fr)
  align(
    center,
    [Направление _02.03.01 «Математика и компьютерные науки»_]
  )
  align(
    center,
    [Основная образовательная программа _СВ.5001.2021 «Математика и компьютерные науки»_],
  )

  v(.2fr)
  align(right, "Научный руководитель: \n профессор кафедры вычислительной математики, \n д.ф.-м.н. Бурова И. Г.")

  v(.1fr)
  align(right, "Рецензент: \n доцент кафедры систем управления ракет СПбГМТУ \n д.ф.-м.н., В. Б. Хазанов.")
  // v(.2fr)
  // align(left, "Работа выполнена полностью, предлагаю поставить зачет")
  v(.5fr)
  pagebreak()

  set page(footer: auto, numbering: "1")

  outline(depth: 3, indent: 1em)
  pagebreak()

  body

  pagebreak()
  heading(level: 1, "Список литературы")
  bibliography(bibliography-file, full: true, title: none, style: "gost-r-705-2008-numeric")

  pagebreak()
  heading(level: 1, "Приложения")
  include "last.typ"
}

#show: doc.with(bibliography-file: "references.bib")

= Введение
Вырожденные дифференциальные уравнения второго порядка представляют собой важный объект исследований как в
теоретическом, так и в прикладном аспектах. Они возникают в различных областях науки и техники, таких как газовая
динамика @L2022, моделирование распространения вязких жидкостей @F1990, @P2023, квантовая космология @H2024 и теория
вероятностей @F2022. Особенность этих уравнений заключается в обращении в ноль коэффициента при старшей производной в
некоторых точках области определения, что влечёт за собой серьёзные трудности при их аналитическом и численном решении
@H2021b.

Исследования, посвящённые этим уравнениям, охватывают широкий круг вопросов, связанных с существованием, единственностью
и асимптотическим поведением решений, а также с особенностями задач Коши и краевых задач. В частности, в работе Н. Х.
Розова, В. Г. Сушко и Д. И. Чудовой @fpm рассматриваются обыкновенные дифференциальные уравнения второго порядка с
коэффициентом при старшей производной, обращающимся в нуль. В ней доказываются теоремы о существовании и единственности
решений краевых задач, уделяется внимание бисингулярным задачам, при которых коэффициенты обращаются в нуль на множестве
положительной меры. Такие задачи требуют применения специальных методов, поскольку стандартные подходы могут оказаться
неприменимыми.

Дополнительный вклад в изучение асимптотического поведения решений внесли В. П. Архипов и А. В. Глушак. В статье "Вырождающиеся
дифференциальные уравнения второго порядка. Асимптотические представления решений" @assymptotics предложены методы
нахождения собственных значений и оценки резольвенты задачи Дирихле. Эти методы позволяют исследовать решения в
комплексной плоскости и по параметру, что делает их универсальными для анализа широкого класса задач. Подход к
построению степенных асимптотик в окрестности точки вырождения разработан в работе "Первые асимптотики решений
вырождающихся дифференциальных уравнений второго порядка" @firstassymptotics, где строятся точные представления решений
с использованием специальных аппроксимаций.

Особое внимание уделено задаче Коши @cauchy, в которой начальные условия задаются в точке вырождения. В. Архипов и А.
Глушак предложили методы построения первых асимптотик решений и показали, что форма начальных условий существенно
зависит от знака коэффициента при первой производной, что оказывает влияние на вид решений и методику их построения.

Современные исследования также сосредоточены на численных методах решения вырожденных задач. Актуальными являются
подходы, основанные на вариационных принципах, разностных схемах и сплайн-аппроксимациях @H2021, @C2020, @N2021.
Локальные сплайны, в частности, обеспечивают высокую точность аппроксимации как самого решения, так и его производных,
особенно вблизи точки вырождения @S2024. Среди них выделяются полиномиальные сплайны второго порядка и сплайны Эрмита
первой высоты, позволяющие получать гладкие приближения, сохраняющие дифференциальные свойства точного решения @S2024b.

Вариационная постановка задачи в подходящем весовом пространстве Соболева @P2022 обеспечивает корректную формулировку
краевой задачи с сильным вырождением. При этом показатель степени $alpha$ в коэффициенте $k(x) = x^alpha p(x)$ лежит в
интервале $[1,2)$, что требует особого подхода к построению граничных условий и численных схем. Разработанный численный
алгоритм сочетает сплайн-аппроксимацию с проекционными методами, что обеспечивает сходимость в энергетической норме и
устойчивость решения. Эффективность предложенного метода подтверждена теоретическим анализом и численными экспериментами
@J2021.

Множество результатов в области разработано в последнее время. В @F2021, @K2021 исследуются условия существования и
единственности решений, а в @H2021 рассматривается введение весовых функций для постановки задач в пространствах
Соболева. В работе @C2020 обсуждаются собственные значения для линеаризованных задач, а в @S2023 уточняется спектральная
структура решений.

Методы, основанные на B-сплайнах и кусочно-линейных функциях, продемонстрировали свою эффективность @M2023, @N2021.
Эрмитовы сплайны @L2023 обеспечивают высокий порядок аппроксимации и позволяют приближать как само решение, так и его
производные @K2022. В @D2022 изучается нестационарное интегро-дифференциальное уравнение с вырожденным оператором, а в
@P2024 предлагается метод локального улучшения аппроксимаций, полученных методом конечных элементов.

Особый интерес представляют случаи с периодическими коэффициентами или решениями, для которых применяются
тригонометрические сплайны @V2022. Вариационные методы оказываются особенно полезны в задачах с неоднородной структурой
или переменными коэффициентами. В @N2021 описан алгоритм адаптивной сетки, позволяющий улучшить точность на участках с
резкими изменениями решений @L2021.

Заключительно, предложенный в настоящей работе подход объединяет достоинства вариационного формулирования и
сплайн-аппроксимации, обеспечивает высокую точность и устойчивость при решении краевых задач с сильным вырождением, а
также демонстрирует широкую применимость в прикладных задачах, связанных с моделированием сложных физических процессов
@R2023b.

// Исследования, посвящённые вырождающимся дифференциальным уравнениям второго порядка, охватывают широкий круг вопросов,
// связанных с существованием, единственностью и асимптотическим поведением решений, а также с особенностями задач Коши и
// краевых задач.

// В работе Н. Х. Розова, В. Г. Сушко и Д. И. Чудовой @fpm изучены обыкновенные дифференциальные уравнения второго порядка
// с коэффициентом при старшей производной, обращающимся в нуль. Доказываются теоремы о существовании и единственности
// решений краевых задач. Важное внимание уделено так называемым бисингулярным задачам, возникающим при вырожденных
// уравнениях, коэффициенты которых обращаются в нуль на некотором множестве. Отмечается, что такие задачи требуют
// дополнительных исследований, поскольку в ряде случаев решения могут отсутствовать в заданном классе функций или требуют
// применения специальных методов для доказательства существования.

// Методы построения асимптотических представлений решений вырождающихся дифференциальных уравнений второго порядка
// подробно рассмотрены в работах В. П. Архипова и А. В. Глушака. В их статье "Вырождающиеся дифференциальные уравнения
// второго порядка. Асимптотические представления решений" @assymptotics предложены формулы для нахождения собственных
// значений, а также оценки резольвенты задачи Дирихле. Эти методы позволяют исследовать решения в комплексной плоскости, а
// также в зависимости от параметра, что делает их универсальными при анализе широкого класса задач.

// Дополнительное внимание к асимптотическому поведению решений уделено в статье "Первые асимптотики решений вырождающихся
// дифференциальных уравнений второго порядка" @firstassymptotics. Здесь предложен подход, позволяющий строить точные
// степенные асимптотики решений в окрестности точки вырождения. Такой метод важен для анализа особенностей поведения
// решений и их сходимости.

// Отдельно рассматривается задача Коши для вырождающихся дифференциальных уравнений второго порядка @cauchy. В. Архипов и
// А. Глушак изучили разрешимость задач Коши с начальными условиями, заданными в точке вырождения, и предложили методы для
// определения первых асимптотик решений. В статье приводятся примеры, иллюстрирующие теоретические положения. Авторами
// установлено, что вид начальных условий существенно зависит от знака коэффициента при первой производной, что
// обуславливает специфику построения решений в данных задачах.

// ---

// Дифференциальные уравнения с сильным вырождением представляют собой важный класс математических задач, возникающих в различных областях науки и техники, таких как газовая динамика @L2022, моделирование распространения вязких жидкостей @F1990, @P2023, квантовая космология @H2024 и теория вероятностей @F2022. Особенностью таких уравнений является обращение в ноль коэффициента при старшей производной в некоторых точках области определения, что приводит к существенным трудностям при их аналитическом и численном решении @H2021b.

// В последние годы активно развиваются методы численного решения вырожденных задач, основанные на вариационных подходах, разностных схемах и сплайн-аппроксимациях @H2021, @C2020, @N2021. Одним из перспективных направлений является применение локальных сплайнов, которые позволяют эффективно аппроксимировать решение и его производные, особенно в случаях, когда традиционные методы оказываются недостаточно точными или устойчивыми @S2024.

// В данной работе рассматривается краевая задача для вырожденного дифференциального уравнения второго порядка, где коэффициент при старшей производной обращается в ноль на границе области, что приводит к существенным особенностям в поведении решения. Такие задачи возникают при моделировании широкого круга физических процессов, включая задачи газовой динамики, квантовой механики и теории фильтрации, где вырождение уравнения отражает специфику физической постановки. Особенностью рассматриваемого класса задач является сильное вырождение, когда показатель степени $alpha$ в коэффициенте $k(x) = x^alpha p(x)$ лежит в интервале $[1,2)$, что приводит к необходимости специального подхода к постановке граничных условий и разработке численных методов. Основной акцент в работе делается на построении эффективного численного алгоритма, сочетающего преимущества сплайн-аппроксимации и вариационного подхода. Использование локальных сплайнов Лагранжа и Эрмита позволяет учесть особенности решения в окрестности точки вырождения, обеспечивая при этом высокий порядок аппроксимации во всей области @A2021. Полиномиальные сплайны второго порядка обеспечивают базовую аппроксимацию решения, в то время как сплайны Эрмита первого уровня дают возможность одновременно приближать как само решение, так и его первую производную, что особенно важно для задач с сильным вырождением @S2024b. Тригонометрические сплайны применяются в случаях, когда свойства решения или коэффициентов уравнения имеют периодический характер @V2022. Вариационная постановка задачи позволяет естественным образом учесть вырожденный характер уравнения, формулируя задачу в подходящем весовом пространстве Соболева @P2022. Предлагаемый подход сочетает в себе преимущества проекционных методов, обеспечивающих сходимость в энергетической норме, и локальных сплайн-аппроксимаций, дающих возможность эффективного вычисления решения в узлах сетки и между ними. Особое внимание уделяется построению аппроксимаций в окрестности точки вырождения, где традиционные методы могут терять точность или устойчивость. Разрабатываемый метод позволяет получать приближенное решение, сохраняющее дифференциальные свойства точного решения, что подтверждается как теоретическим анализом, так и результатами численных экспериментов @J2021. Важным аспектом исследования является сравнение эффективности полиномиальных и тригонометрических сплайнов для различных типов правых частей и коэффициентов уравнения, что позволяет выработать рекомендации по выбору типа аппроксимации в зависимости от особенностей конкретной задачи @S2021. Теоретическое обоснование метода включает оценку погрешности аппроксимации в весовых нормах, учитывающих вырожденный характер уравнения, а также анализ устойчивости и сходимости построенной разностной схемы @F2023. Практическая значимость работы заключается в том, что предлагаемый алгоритм может быть эффективно реализован для решения широкого класса прикладных задач, описываемых вырожденными уравнениями, и допускает различные модификации, такие как использование неравномерных сеток или адаптивных алгоритмов для повышения точности в областях с большими градиентами решения @R2023b.

// Обзор литературы. Вырожденные дифференциальные уравнения изучаются в работах многих авторов. В @F2021, @K2021 исследуются вопросы существования и единственности решений для различных классов вырожденных задач. В @H2021 предложен подход, основанный на введении специальных весовых функций, обеспечивающих корректную постановку задачи в пространствах Соболева. В @C2020 рассматриваются условия существования собственных значений для линеаризованных вырожденных уравнений @S2023.

// Численные методы решения вырожденных задач активно развиваются. В @O2021 изучается вырожденное параболическое уравнение, моделирующее пространственную диффузию биологических популяций, а в @P2024 рассматриваются задачи фильтрации газа. Для аппроксимации решений часто используются B-сплайны и кусочно-линейные функции @M2023, @N2021. В @S2024 предложены сплайны Эрмитова типа @L2023, обеспечивающие высокий порядок аппроксимации и возможность одновременного вычисления решения и его производных @K2022.

// Вариационные методы и методы конечных элементов также находят применение при решении вырожденных задач. В @D2022 исследуется нестационарное интегро-дифференциальное уравнение с вырожденным эллиптическим оператором, а в @P2024 предложен метод локального улучшения приближенного решения, полученного методом конечных элементов. В @N2021 разработан алгоритм адаптивной сетки для одномерных краевых задач второго порядка @L2021.

// Особый интерес представляют методы, основанные на сплайн-аппроксимации. В @P2024 локальные сплайны применяются в методе наименьших квадратов для решения краевых задач. В настоящей работе развивается этот подход, предлагается использование сплайнов Эрмитова типа первого уровня, что позволяет получать непрерывно дифференцируемые приближения решения и его производных.

// ---

// Такие вырождающиеся дифференциальные уравнения возникают при решении многих прикладных задач. Кроме указаных в @burova
// задачах динамики воздушных масс gas, растекания вязкой жидкости по плоской поверхности liquid1, liquid2, такие уравнения
// возникают также в задачах теплопроводности в композитных материалах composite, течении жидкостей сквозь пористные
// материалы porous, эластичности и прочности материалов elasticity.

// Среди новых работ отметим admissible, dong.

// В admissible были разработаны теоремы регулярности для вырождающихся эллиптических уравнений, где вырождение подчинено
// весу $Phi$. В том числе были найдены достаточные условия для локальной ограниченности обобщенных решений, а также
// найдены достаточные условия непрерывности решения.

// В dong авторы нашли достаточные условия на веса, при которых есть существование, единственность и регулярность решений в
// пространствах Соболева. Эта работа -- первая с такими результатами и нова даже для случая постоянных коэффициентов.

// Более остальных нас интересуют краевые задачи для вырождающихся эллиптических уравнений. Такие задачи возникают в
// различных прикладных областях, где поведение системы качественно меняется из-за вариации определенных параметров или
// условий.

// В финансовой математике вырожденные эллиптические операторы возникают при моделировании стохастических процессов
// волатильности активов. Вырожденность связана с условиями рынка, когда волатильность активов стремится к нулю volatility.

// В механике такие уравнения возникают в ситуациях частично расплавленых материалов. Например в земной мантии и ледниках
// mixtures.

// Таким образом, при решении многих задач моделирования физических процессов, часто приходится численно решать краевые или
// начальные задачи эллиптических дифференциальных уравнений с вырождениями.

= Цель работы
Разработать и исследовать методы аппроксимации решений краевых задач для вырождающихся дифференциальных уравнений
второго порядка. В частности, основное внимание уделяется:

1. Формализации аппроксимации в функциональных пространствах, таких как $WW_2^1(0,1)$, с использованием
  вариационно-сеточных методов.
2. Разработке и анализу координатных систем для одномерных задач, обеспечивающих сходимость аппроксимации в энергетической
  норме.
3. Оценке порядка аппроксимации, полученной с использованием линейной интерполяции, а также практическому применению
  методов к конкретным задачам.
4. Построению численных методов и их верификации на тестовых примерах с известными аналитическими решениями.

Целью является не только теоретический анализ разработанных методов, но и их практическое применение, что позволит
подтвердить эффективность предложенных подходов.

= Аппроксимация решения первой краевой задачи для вырождающихся одномерных дифференциальных уравнений второго порядка

== Вид уравнения
В данной главе работы будет рассмотрено уравнение следующего вида:
$
  -d/dx [x^alpha p(x) du/dx] + q(x) u = f(x), quad 0 < x < 1, quad f in L2(0, 1),\
  q "измерима, ограничена, неотрицательна на" [0,1]\
  alpha = const > 0, space p in C^1 [0,1], space p(x) >= p_0 = const > 0\
  u in WW_2^1(0,1)
$
== Аппроксимация в пространстве $WW_2^1$ <2.2>
=== Основа вариацонно-сеточного метода
Пусть в гильбертовом пространстве $H$ действует линейный положительно-определенный оператор $A$ и требуется найти
решение уравнения
$
  A u = f, quad f in H
$
Принято вводить функционал энергии и энергетическую норму с энергетическим произведением:
$
  [u,v]_A := (A u,v),\
  norm(u)_A^2 := [u,u]_A = (A u,u),\
  cal(F)(u) := 1/2[u,u]_A - (f,u)
$
При положительной определенности оператора $A$ функционал энергии является выпуклым, из чего следует, что ноль его
производной является точкой минимума.
$
  cal(F)'(u)h = [u,h]_A - (f,h)\
  cal(F)'(u_0) = 0 <=> forall h in H quad [u_0,h]_A = (f,h) <=> A u_0 = f\
$
Если решение $u_0$ аппроксимируется в конечномерном пространстве $H_n$ с энергетическим произведением $[dot,dot]_A$, то
критерием минимальности функционала энергии в точке $u_n in H_n$ будет следующая система линейных уравнений:
$
  u_n = sum_(k=1)^n a_k phi_k, " где Lin" {phi_1,dots,phi_n} = H_n,\
  (cal(F) bar_H_n) '(u_n) = 0 <=> partial/(partial a_k) cal(F)(u_n) = 0, space k = 1,...,n\
  partial/(partial a_k) cal(F)(u_n) = partial/(partial a_k) [1/2 sum_(i=1)^n sum_(k=1)^n a_i a_k [phi_i, phi_k]_A - sum_(i=1)^n a_i (f, phi_i)] =\
  = sum_(i=1)^n a_i [phi_i, phi_k]_A - (f, phi_k) =0, space k = 1,...,n\
$<system>
Приведенная схема называется методом Ритца.

Основой сеточного метода аппроксимации является выбор функций $phi_i$, которые связаны с координатной сеткой в области
аппроксимации и задаются простыми формулами. Эти базисные функции $phi_i$ мы будем называть координатными функциями.
Выбор функций ограничен лишь условием полноты системы ${{phi_(n, i)}_(i=1)^k_n}_n$, где для каждого $n$ задается
подпространство $H_n$ размерности $k_n$ и функции $phi_(n, i)$ образуют базис в этом подпространстве, а полнота системы
-- это условие
$
  forall u in H space lim_(n -> oo) inf_(v_n in H_n) norm(u - v_n)_A = 0,
$
то есть любая функция $u$ может быть аппроксимирована с любой точностью в энергетической норме.

В книге @michbook показано, что если координатная система ${phi_(n, i)}$ полна в смысле описанном выше, то построенная
при помощи системы @system аппроксимация сходится в энергетической норме к решению исходного уравнения.

Р. Курантом в @kurant было показано, что необязательно выбирать последовательность подпространств $H_n$, которые строго
вложены друг в друга, как изначально предполагалось в методе Ритца, главное, чтобы система базисов этих пространств была
полна.

== Одномерный случай
Вернемся к поставленной задаче, в @methods описаны необходимые и достаточные условия для минимальной координатной
системы в $WW_p^s (Omega subset RR^m)$ вида
$
  {phi_(q, j, h) (x) = omega_q (x/h - j)}_(j in J_h, abs(q) = 0, ..., s-1),
$
которая при помощи функции $u_h$ аппроксимирует любую функцию $u in C_0^s (overline(Omega))$ в метрике $circle(WW)_p^s (Omega)$,
а также эту же функцию $u$ в метрике $C^(s-1) (K)$ для любого компакта $K subset Omega$ при $h -> 0$. Здесь $h$ -- шаг
сетки, $J_h$ -- конечный набор целых мультииндексов размера $m$, такой что $union.big_(j in J_h) "supp" phi_(q, j, h) supset Omega space forall q$,
а аппроксимирующая функция $u_h$ определяется как
$
  sum_(abs(q) = 0)^(s-1) sum_(j in J_h) h^q u^((q))((j+bb(1))h) omega_q (x/h-j)
$

Рассмотрим одномерный случай, то есть $m = 1$ и частный пример $Omega = (0,1)$. Для данного случая в @methods подробно
описаны рекурсивные формулы для построения функций $omega_q$. Но эти же функции можно получить рассмотрев более простую
задачу -- построение полиномиальной координатной системы.

== Полиномиальные координатные системы
Будем использовать общепринятый сеточный метод аппроксимации. Начнем с отрезка $[0,2]$ и кусочных функций на нем. Пусть
$
  omega_q = cases(phi_q (x) &", " x in [0,1], psi_q (x-1) &", " x in [1,2], 0 &", " x in.not [0,2]), quad psi_q, phi_q : [0,1] -> RR "полиномы", space q = 0,...,s-1 \
  phi_q (0) = 0, space psi_q (1) = 0, space phi_q (1) = psi_q (0), quad q = 0,...,s-1
$

Теперь рассмотрим сдвиги функций $omega_q$:
$
  {omega_(j, q) := omega_q (x-j)}_(j in ZZ, q = 0,...,s-1)
$
Носители этих функций -- отрезки $[j, j+2]$. Теперь построим приближение функции $u in C^(s-1) ([0,2])$ в виде
$
  tilde(u) (x) = sum_(j in ZZ inter [0,2]) sum_(q=0)^(s-1) u^((q)) (j) omega_(j, q) (x), quad x in [0,2]
$
Рассмотрим отрезок $[0,1]$, тогда в формуле выше записан некий полином степени $s-1$ с коэффициентами $u^((q)) (j)$.
Чтобы найти функции $phi_q, psi_q$ необходимо предположить, что для $u = 1,x,...,x^r$ выполняется равенство $tilde(u) = u$.
Это условие дает систему линейных уравнений для коэффициентов $phi_q, psi_q$. Понятно, что стоит рассматривать $r >= s-1$,
так как иначе последние производные будут равны нулю и функцию $omega_(s-1)$ мы не найдем. Начиная с некоторого $r$ эта
система будет иметь единственное решение, которое и будет определять функции $omega_q$.

Приведу решения для $s = 1,2,3$:

1. $s = 1$
$
  omega_0 (x) = cases(x &", " x in [0,1], 2-x &", " x in [1,2], 0 &", " x in.not [0,2])
$
#pagebreak()
2. $s = 2$
$
    &omega_0 (x) = cases(-2x^3 + 3x^2 &", " x in [0,1], 2x^3 - 9x^2 + 12x - 4 &", " x in [1,2], 0 &", " x in.not [0,2]) \
    &omega_1 (x) = cases(x^3 - x^2 &", " x in [0,1], x^3 - 5x^2 + 8x - 4 &", " x in [1,2], 0 &", " x in.not [0,2])
$

3. $s = 3$
$
    &omega_0 (x) = cases(
    6x^5 - 15x^4 + 10x^3 &", " x in [0,1],
    -6x^5 + 45x^4 - 130x^3 + 180^2 - 120x + 32 &", " x in [1,2],
    0 &", " x in.not [0,2],

  ) \
    &omega_1 (x) = cases(
    -3x^5 + 7x^4 - 4x^3 &", " x in [0,1],
    -3x^5 + 23x^4 - 68x^3 + 96x^2 -64x + 16 &", " x in [1,2],
    0 &", " x in.not [0,2],

  ) \
    &omega_2 (x) = cases(
    1/2x^5 - x^4 + 1/2x^3 &", " x in [0,1],
    -1/2x^5 + 4x^4 - 25/2x^3 + 19x^2 - 14x + 4 &", " x in [1,2],
    0 &", " x in.not [0,2],

  )
$

Далее, следуя теории из @methods, добавляем шаг сетки, строим систему ${phi_(q, j, h)}_(j in ZZ, q = 0,...,s-1)$ и
аппроксимируем функцию $u in C^(s-1) (0,1)$.

== Вид системы при различных высотах $s$
В @system для нахождения коэффициентов $a_i$ приводится явный вид системы линейных уравнений. В случае $s > 0$ систему
можно привести к простому виду матричного уравнения. Рассмотрим например $s = 1$.

$
  u_n = sum_(j = 1)^n (a_(j, 0) omega_(j, 0) + a_(j, 1) omega_(j, 1)) \
  partial/(partial a_(j, 0)) cal(F)(u_n) = partial/(partial a_(j, 1)) cal(F)(u_n) = 0, space j = 1,...,n
$

Можно выделить 4 вида взаимодействия между базисными функциями $omega_(j, q)$:
$
    &M_(0, 0) = ([omega_(j, 0), omega_(i, 0)]_A)_(i, j) quad quad quad
    &&M_(0, 1) = ([omega_(j, 0), omega_(i, 1)]_A)_(i, j) \
    &M_(1, 0) = ([omega_(j, 1), omega_(i, 0)]_A)_(i, j) quad quad quad
    &&M_(1, 1) = ([omega_(j, 1), omega_(i, 1)]_A)_(i, j) \
$

А значит представить всю систему в виде матричного уравнения
$
  mat(M_(0, 0), M_(0, 1);M_(1, 0), M_(1, 1)) vec(a_0, a_1) = vec((f, omega_0), (f, omega_1))
$

Так как носители функций $omega_(j, q)$ -- отрезки $[j, j+2]$, то матрицы $M_(0 0), M_(0 1), M_(1 0), M_(1 1)$ будут
иметь трехдиагональный вид, что упрощает решение.

Аналогично для остальных случаев $s > 1$ систему можно свести к матричному уравнению
$
  mat(
    M_(0, 0), M_(0, 1), ..., M_(0, s-1);M_(1, 0), M_(1, 1), ..., M_(1, s-1);dots.v, dots.v, dots.down, dots.v;M_(s-1, 0), M_(s-1, 1), ..., M_(s-1, s-1)
  ) vec(a_0, a_1, ..., a_(s-1)) = vec((f, omega_0), (f, omega_1), ..., (f, omega_(s-1)))
$

== Порядок аппроксимации
Вернемся к нашей задаче, одномерный случай, $s = 1$, $Omega = (0,1)$. Координатная система в этом случае имеет вид
$
  {phi_(j, h) (x) = omega (x/h - j)}_(j in J_h)
$
А аппроксимирующая функция имеет вид
$
  u_h (x) = sum_(j in J_h) u((j+1)h) omega (x/h - j)
$
Для удобства введем обозначение $x_j := j h$. Тогда на промежутке $[x_(j), x_(j+1)]$ ненулевыми $phi_(dot, h)$ будут
только $phi_(j, h)$ и $phi_(j-1, h)$.

Тогда на этом промежутке
$
  u_h (x) = u(x_(j+1)) phi_(j, h) (x) + u(x_j) phi_(j-1, h) (x),
$
но на каждом промежутке $[x_j, x_(j+1)]$ $phi_(dot, h)$ являются полиномами степени не выше 1, и верно $phi_(i-1,h) (x_i) = 1$,
то есть на самом деле мы имеем дело с линейной интерполяцией: вписываем ломаную в график функции $u$ в точках $x_j$.

Для интерполяционного многочлена есть оценка оценка остатка:
$
  x in [x_j, x_(j+1)] =>\
  abs(u(x) - u_h (x)) <= sup_(x in (x_j, x_(j+1))) abs(u''(x)) dot 1 / 2! abs((x-x_j)(x-x_(j+1)))
$
Несложно проверить, что $abs((x-x_j)(x-x_(j+1))) <= 1/4 h^2$, тогда
$
  norm(u-u_h)_C(0,1) <= 1/4 h^2 sup_(x in (0,1)) abs(u''(x))
$
То есть мы ожидаем не лучше, чем квадратичную сходимость.

== Погрешность приближения
В предыдущем пункте была получена оценка погрешности приближения $u$ линейной интерполяцией. В @gusman получена важная
для нас оценка:
$
  norm(u - u^h)_A <= norm(u - u_h)_A,
$
где $u^h$ -- приближенное решение, получаемое при помощи метода Ритца, а $u_h$ -- вписанная в график $u$ ломаная, с
узлами в точках $x_j$. Таким образом, при наличии оценки на $norm(u - u_h)_A$, можно оценить погрешность приближения $u^h$ к $u$.

=== Случай слабого вырождения
Случай $0 < alpha < 1$ называется случаем слабого вырождения. В данном случае в @methods было показано, что
$
  norm(u - u_h)_A <= C norm(f)_L_2 h^((1-alpha) slash 2).
$
И данная оценка точна в том смысле, что существует функция $u$, для которой $norm(u - u_h)_A = C norm(f)_L_2 h^((1-alpha) slash 2)$.

=== Случай сильного вырождения
Случай $1 <= alpha < 2$ называется случаем сильного вырождения. В данном случае в @methods было показано, что
$
  norm(u - u_h)_A <= C norm(f)_L_2 h^(1 - alpha/2).
$
И данная оценка почти точна в том смысле, что для любого $epsilon > 0$ существует функция $u$, для которой $norm(u - u_h)_A >= C h^(1 - alpha/2 - epsilon)$.

==== Улучшение оценки
На самом деле (см. @methods), при $f in L_r (0,1), space 2 < r <= oo$, можно получить лучшую оценку в случае $1 <= alpha < 2$:
$
  norm(u - u_h)_A <= C norm(f)_L_r h^((3 - alpha)/2 - 1/r),
$
где при $r = oo$ следует считать $1/r = 0$.

== Применение к конкретным задачам

=== Аппроксимация сплайнами нулевой высоты
В качестве первого примера рассмотрим задачу
$
  -d /(dx) (x^alpha du/dx) + u = (3^(3-alpha)-2(3-alpha)x-1)/(3-alpha),\
  0 < x < 1 quad 1 <= alpha <= 2
$
Известно, что в данном случае нужно ставить условие только на конце $u(1)$. Пусть $u(1) = 0$. Правая часть уравнения
получалась подстановкой
$
  u(x) = (x^(3-alpha)-1)/(3-alpha),
$
то есть это точное решение задачи.

Возьмем натуральное $n$ и по нему построим $h = 1 slash (n+1)$, тогда в координатной системе ${phi_(i,h)}$ первая и
последняя функции выглядят следующим образом:

#align(
  center,
  cetz.canvas(
    {
      import cetz.draw: *

      let wf = x => calc.max(1 - calc.abs(x - 1), 0)
      let n = 10
      let h = 1 / (n + 1)

      let phi_first = x => wf(x / h + 1)
      let phi_last = x => wf(x / h - n)

      plot.plot(size: (12, 8), x-tick-step: 0.25, y-tick-step: 0.5, y-min: -0.25, y-max: 1.75, legend: "inner-north", {
        let domain = (-0.25, 1.25)
        plot.add(phi_first, domain: domain, samples: 1000, label: [$phi_(-1,h)$])
        plot.add(phi_last, domain: domain, samples: 1000, label: [$phi_(n,h)$])
      })
    },
  ),
)
Будем строить аппроксимирующую функцию вида $u_h = sum_(k=-1)^n a_k phi_(k,h)$, заметим, что так как $u(1) = 0$, то
коэффициент при последнем члене $a_n = 0$.

По предложенному в @2.2[пункте] методу строим аппроксимирующую путем решения системы
$
  sum_(k=-1)^(n-1) a_k [phi_(k,h), phi_(j,h)]_A = (f, phi_(j,h)), quad j = -1,...,n
$
Тут
$
  A u = -d /(dx) (x^alpha du/dx) + u,\
  [phi_(k,h), phi_(j,h)]_A = (A phi_(k,h), phi_(j,h)) =\ = integral_0^1 {(-d /(dx) (x^alpha d/dx phi_(k,h)) + phi_(k,h)) dot phi_(j,h)} dx =\
  = integral_0^1 {x^alpha (d/dx phi_(k,h)) (d/dx phi_(j,h)) + phi_(k,h) phi_(j,h)} dx - lr(x^alpha (d/dx phi_(k,h)) phi_(j,h) |)_0^1
$
Видно, что подстановка $lr(x^alpha (d/dx phi_(k,h)) phi_(j,h) |)_0^1$ равна нулю в случае, если хотя бы один из $k$ или $j$ не
равен $n$. Так как $k = -1,...,n-1$, то подстановка всегда равна $0$. Произведение в правой части -- классическое
скалярное произведение функций
$
  (f, phi_(j,h)) = integral_0^1 f(x) phi_(j,h) (x)
$

Расчеты производились в системе математических вычислений с точностью 8 знаков после запятой.

Результаты вычислений приведены в виде таблиц абсолютных погрешностей

#show table.cell.where(y: 0): strong
#set table(stroke: (x, y) => if y == 0 {
  (bottom: 0.7pt + black)
}, align: (x, y) => (if x > 0 { center } else { left }))

#tables.t1
#tables.t2

Также приведены графики приближенного решения и абсолютных погрешностей для $alpha = 1$:

#grid(
  columns: (1fr, 1fr),
  column-gutter: 3em,
  row-gutter: 2em,
  align(right, figure(plots.p1, caption: [Приближенное решение], supplement: [Гр.])),
  figure(plots.p2, caption: [Абсолютная погрешность], supplement: [Гр.]),
)

И абсолютные погрешности при разных $h$:
#grid(columns: (1fr, 1fr), column-gutter: 3em, row-gutter: 2em, align(right, figure(plots.p3)), figure(plots.p4))

=== Аппроксимация сплайнами первой высоты
В качестве второго примера рассмотрим задачу
$
  -d / (d x) (x^alpha du/dx) + u = 3 (x - 1) x^(alpha - 1) ((alpha + 2) x - alpha) - (x-1)^3, \
  0 < x < 1, quad 1 <= alpha < 2, quad u(1) = u'(1) = 0
$
с аналитическим решением $u(x) = (1 - x)^3$.

Построим базисные функции $omega_0, omega_1$:
#grid(columns: (1fr, 1fr), plots.p5, plots.p6)

В данном случае построение аппроксимирующего решения является совмещением двух этапов:
+ аппроксимируем значения функции-решения и значения ее производной в точках сетки ${x_j}_(j=0)^(n-1)$
+ вписываем базисные функции в полученый каркас приближенного решения
$
  u_h = sum_(j=0)^(n-1) tilde(u)(x_j) omega_(j,0) + h dot tilde(u)'(x_j) omega_(j,1)
$

Формулы подсчета энергетического произведения остаются теми же, что и в случае нулевой высоты.

Результаты вычислений приведены в виде таблиц абсолютных погрешностей

#grid(columns: (1fr, 1fr), tables.t3, tables.t4)

Также приведены графики приближенного решения и абсолютных погрешностей для $alpha = 1.5$:
#figure(
  grid(columns: (1fr, 1fr), column-gutter: 3em, row-gutter: 2em, align(right, plots.p7), plots.p8),
  caption: [Приближенное решение и абсолютная погрешность],
  supplement: [Гр.],
)
#figure(
  grid(columns: (1fr, 1fr), column-gutter: 3em, row-gutter: 2em, align(right, plots.p9), plots.p10),
  caption: [Приближенная производная решения и абсолютная погрешность],
  supplement: [Гр.],
)

// В качестве минуса такого подхода можно отметить большую погрешность в начале отрезка, где решение не закреплено. Так как
// у нас нет информации о поведении слева от точки 0, то мы не можем также точно построить каркас, как в середине отрезка.

// === Задача о двух закрепленных концах
// Рассмотрим более подходящий вариант задачи для аппроксимации сплайнами первого уровня.
// $
//   -d / (d x) (x^alpha du/dx) + u = (x-1)^2 x^(3-alpha) + 2x (2 x^2 (alpha-5) - 3 x (alpha-4) + alpha - 3), \
//   0 < x < 1, quad 1 <= alpha < 2, quad u(0) = u(1) = u'(0) = u'(1) = 0
// $
// с аналитическим решением $u(x) = x ^ (3 - alpha) (1 - x) ^ 2$.

// Графики приближенного решения и абсолютных погрешностей для $alpha = 1.5$:
// #figure(
//   grid(columns: (1fr, 1fr), column-gutter: 3em, row-gutter: 2em, align(right, plots.p11), plots.p12),
//   caption: [Приближенное решение и абсолютная погрешность],
//   supplement: [Гр.],
// )
// #figure(
//   grid(columns: (1fr, 1fr), column-gutter: 3em, row-gutter: 2em, align(right, plots.p13), plots.p14),
//   caption: [Приближенная производная решения и абсолютная погрешность],
//   supplement: [Гр.],
// )

// #pagebreak()
// Таблица абсолютных погрешностей
// #tables.t5

= Заключение
В ходе работы был исследован метод аппроксимации решений краевых задач для вырождающихся дифференциальных уравнений
второго порядка, основанный на вариационно-сеточных методах. В процессе реализации выделены основные этапы аппроксимации
в функциональных пространствах, таких как $WW_2^1(0,1)$, с использованием линейных базисных функций, что позволило
обеспечить сходимость решений в энергетической норме.

Показано, что использование предложенной координатной системы позволяет с высокой точностью аппроксимировать решения в
рамках заданной задачи. Анализ порядка аппроксимации показал, что для линейной интерполяции ошибки аппроксимации
оцениваются квадратично. Верификация полученных численных решений на тестовых примерах с известными аналитическими
решениями подтвердила корректность методов и их применимость к реальным задачам.

Практическое применение метода было продемонстрировано на конкретной задаче с аналитическим решением, где была построена
аппроксимирующая функция, что позволило провести численный эксперимент и получить точные результаты. Таким образом,
результаты работы подтверждают эффективность предложенной аппроксимации для решения краевых задач вырождающихся
дифференциальных уравнений второго порядка, а также подчеркивают важность выбора правильной координатной системы для
достижения необходимой точности.

Выделим особо перспективные приложения данного метода:
+ *Решение задач с вырождением в краевых точках*: предложенный метод эффективно работает при сильном вырождении
  коэффициента при старшей производной, что позволяет решать задачи, моделирующие, например, поведение жидкостей с нулевой
  вязкостью на границе или проблемы теплопереноса в средах с резко неоднородной теплопроводностью.

+ *Моделирование физических процессов в ограниченных геометриях*: за счёт локальности и высокой точности аппроксимации
  производных метод может применяться при численном моделировании задач в цилиндрических или сферических координатах, где
  коэффициенты уравнения вырождаются на оси или границе сферы.

+ *Численный анализ спектральных задач*: за счёт малой погрешности в аппроксимации производной возможно более точное
  приближение собственных значений и собственных функций вырожденных операторов.

+ *Адаптивные схемы на неравномерных сетках*: локальные свойства метода делают его перспективным для использования в
  адаптивных методах с локальным сгущением сетки в окрестности точки вырождения без ухудшения стабильности аппроксимации.

+ *Интегро-дифференциальные уравнения с вырожденными ядрами*: конструкция метода допускает его расширение на классы задач,
  где вырождение затрагивает не только дифференциальную, но и интегральную часть оператора.

+ *Разработка библиотек численного решения*, в частности внедрение метода в широкоиспользуемую библиотеку `scipy`.
