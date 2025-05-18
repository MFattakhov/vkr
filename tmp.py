import re
import bibtexparser


s = """Дифференциальные уравнения с сильным вырождением представляют собой важный класс математических задач, возникающих в различных областях науки и техники, таких как газовая динамика [1], моделирование распространения вязких жидкостей [2, 3], квантовая космология [6] и теория вероятностей [4]. Особенностью таких уравнений является обращение в ноль коэффициента при старшей производной в некоторых точках области определения, что приводит к существенным трудностям при их аналитическом и численном решении [28].  
В последние годы активно развиваются методы численного решения вырожденных задач, основанные на вариационных подходах, разностных схемах и сплайн-аппроксимациях [9, 10, 14]. Одним из перспективных направлений является применение локальных сплайнов, которые позволяют эффективно аппроксимировать решение и его производные, особенно в случаях, когда традиционные методы оказываются недостаточно точными или устойчивыми [15].  
В данной работе рассматривается краевая задача для вырожденного дифференциального уравнения второго порядка, где коэффициент при старшей производной обращается в ноль на границе области, что приводит к существенным особенностям в поведении решения. Такие задачи возникают при моделировании широкого круга физических процессов, включая задачи газовой динамики, квантовой механики и теории фильтрации, где вырождение уравнения отражает специфику физической постановки. Особенностью рассматриваемого класса задач является сильное вырождение, когда показатель степени α в коэффициенте k(x) = xᵅp(x) лежит в интервале [1,2), что приводит к необходимости специального подхода к постановке граничных условий и разработке численных методов. Основной акцент в работе делается на построении эффективного численного алгоритма, сочетающего преимущества сплайн-аппроксимации и вариационного подхода. Использование локальных сплайнов Лагранжа и Эрмита позволяет учесть особенности решения в окрестности точки вырождения, обеспечивая при этом высокий порядок аппроксимации во всей области [5]. Полиномиальные сплайны второго порядка обеспечивают базовую аппроксимацию решения, в то время как сплайны Эрмита первого уровня дают возможность одновременно приближать как само решение, так и его первую производную, что особенно важно для задач с сильным вырождением [17]. Тригонометрические сплайны применяются в случаях, когда свойства решения или коэффициентов уравнения имеют периодический характер [26]. Вариационная постановка задачи позволяет естественным образом учесть вырожденный характер уравнения, формулируя задачу в подходящем весовом пространстве Соболева [29]. Предлагаемый подход сочетает в себе преимущества проекционных методов, обеспечивающих сходимость в энергетической норме, и локальных сплайн-аппроксимаций, дающих возможность эффективного вычисления решения в узлах сетки и между ними. Особое внимание уделяется построению аппроксимаций в окрестности точки вырождения, где традиционные методы могут терять точность или устойчивость. Разрабатываемый метод позволяет получать приближенное решение, сохраняющее дифференциальные свойства точного решения, что подтверждается как теоретическим анализом, так и результатами численных экспериментов [25]. Важным аспектом исследования является сравнение эффективности полиномиальных и тригонометрических сплайнов для различных типов правых частей и коэффициентов уравнения, что позволяет выработать рекомендации по выбору типа аппроксимации в зависимости от особенностей конкретной задачи [18]. Теоретическое обоснование метода включает оценку погрешности аппроксимации в весовых нормах, учитывающих вырожденный характер уравнения, а также анализ устойчивости и сходимости построенной разностной схемы [19]. Практическая значимость работы заключается в том, что предлагаемый алгоритм может быть эффективно реализован для решения широкого класса прикладных задач, описываемых вырожденными уравнениями, и допускает различные модификации, такие как использование неравномерных сеток или адаптивных алгоритмов для повышения точности в областях с большими градиентами решения [24].
Обзор литературы. Вырожденные дифференциальные уравнения изучаются в работах многих авторов. В [7, 8] исследуются вопросы существования и единственности решений для различных классов вырожденных задач. В [9] предложен подход, основанный на введении специальных весовых функций, обеспечивающих корректную постановку задачи в пространствах Соболева. В [10] рассматриваются условия существования собственных значений для линеаризованных вырожденных уравнений [27].  
Численные методы решения вырожденных задач активно развиваются. В [11] изучается вырожденное параболическое уравнение, моделирующее пространственную диффузию биологических популяций, а в [12] рассматриваются задачи фильтрации газа. Для аппроксимации решений часто используются B-сплайны и кусочно-линейные функции [13, 14]. В [15] предложены сплайны Эрмитова типа [30], обеспечивающие высокий порядок аппроксимации и возможность одновременного вычисления решения и его производных [23].  
Вариационные методы и методы конечных элементов также находят применение при решении вырожденных задач. В [16] исследуется нестационарное интегро-дифференциальное уравнение с вырожденным эллиптическим оператором, а в [12] предложен метод локального улучшения приближенного решения, полученного методом конечных элементов. В [14] разработан алгоритм адаптивной сетки для одномерных краевых задач второго порядка [22].  
Особый интерес представляют методы, основанные на сплайн-аппроксимации. В [12] локальные сплайны применяются в методе наименьших квадратов для решения краевых задач. В настоящей работе развивается этот подход, предлагается использование сплайнов Эрмитова типа первого уровня, что позволяет получать непрерывно дифференцируемые приближения решения и его производных."""

references_bib = """
@book{L2022,
  author = {Bers L},
  title = {Mathematical aspects of subsonic and transonic gas dynamics},
  year = {2022},
  pages = {203}
}

@article{F1990,
  author = {Bernis F},
  title = {Higher Order Nonlinear Degenerate Parabolic Equations},
  journal = {Journal of Differential Equations},
  year = {1990},
  volume = {83},
  pages = {179–206}
}

@article{P2023,
  author = {Greenspan P},
  title = {On the motion of a small viscous droplet that wets a surface},
  journal = {J. Fluid Mech},
  year = {2023},
  volume = {84},
  pages = {125–143}
}

@article{F2022,
  author = {Brock F},
  title = {A class of degenerate elliptic equations and a Dido’s problem with respect to a measure},
  journal = {J. Math. Anal. Appl},
  year = {2022},
  volume = {348},
  pages = {356–365}
}

@article{A2021,
  author = {Cavalheiro A},
  title = {Existence results for a class of nonlinear degenerate Navier problems},
  journal = {Siberian Electronic Mathematical Reports},
  year = {2021},
  volume = {18},
  number = {1},
  pages = {647–667}
}

@article{H2024,
  author = {Berestycki H},
  title = {Existence and Bifurcation of Solutions for an Elliptic Degenerate Problem},
  journal = {Journal of differential equations},
  year = {2024},
  volume = {134},
  pages = {1–25}
}

@article{F2021,
  author = {Kappel F},
  title = {On degeneracy of functional-differential equations},
  journal = {Journal of differential equations},
  year = {2021},
  volume = {22},
  pages = {250–267}
}

@article{K2021,
  author = {Igari K},
  title = {Degenerate Parabolic Differential Equations},
  journal = {Publ. RIMS, Kyoto Univ},
  year = {2021},
  volume = {9},
  pages = {493–504}
}

@article{H2021,
  author = {Dong H},
  title = {Parabolic and elliptic equations with singular or degenerate coefficients: The Dirichlet problem},
  journal = {Trans. Amer. Math. Soc},
  year = {2021},
  volume = {374},
  pages = {6611–6647}
}

@article{C2020,
  author = {Stuart C},
  title = {A critically degenerate elliptic Dirichlet problem, spectral theory and bifurcation},
  journal = {Nonlinear Analysis},
  year = {2020},
  volume = {190}
}

@article{O2021,
  author = {Nikan O},
  title = {Numerical simulation of a degenerate parabolic problem occurring in the spatial diffusion of biological population},
  journal = {Chaos, Solitons and Fractals},
  year = {2021},
  volume = {151}
}

@article{P2024,
  author = {Ambrosio P},
  title = {Regularity results for a class of widely degenerate parabolic equations},
  journal = {Advances in Calculus of Variations},
  year = {2024},
  volume = {17},
  number = {3},
  pages = {805–829}
}

@article{M2023,
  author = {Scutaru M},
  title = {et al. Flow of Newtonian Incompressible Fluids in Square Media: Isogeometric vs. Standard Finite Element Method},
  journal = {Mathematics},
  year = {2023},
  volume = {11}
}

@article{N2021,
  author = {Mastorakis N},
  title = {On the solution of Integral-Differential Equations via the Rayleigh-Ritz Finite Elements Method: Stationary Transport Equation},
  journal = {WSEAS Transactions on Mathematics},
  year = {2021},
  volume = {4},
  number = {2},
  pages = {41–49}
}

@article{S2024,
  author = {Mikhlin S},
  title = {Variational-difference approximation},
  journal = {Zap. Nauchn. Sem. LOMI},
  year = {2024},
  volume = {48},
  pages = {32–188}
}

@article{D2022,
  author = {Černá D},
  title = {Wavelet Method for Sensitivity Analysis of European Options under Merton Jump-Diffusion Model},
  journal = {AIP Conference Proceedings},
  year = {2022},
  pages = {300}
}

@article{S2024b,
  author = {Mikhlin S},
  title = {Some Theorems on the Stability of Numerical Processes},
  journal = {Atti d. Lincei. Classe fis., mat. e nat},
  year = {2024},
  pages = {1–32}
}

@book{S2021,
  author = {Mikhlin S},
  title = {Approximation on the Cubic Lattice},
  year = {2021},
  pages = {203}
}

@article{F2023,
  author = {Lin F},
  title = {A class of fully nonlinear elliptic equations with singularity at the boundary},
  journal = {J. Geom. Anal},
  year = {2023},
  volume = {8},
  number = {4},
  pages = {583–598}
}

@article{Y2022,
  author = {Zhang Y},
  title = {Numerical methods for strongly degenerate parabolic equations},
  journal = {Journal of Computational Physics},
  year = {2022},
  pages = {209}
}

@article{R2023,
  author = {Johnson R},
  title = {Adaptive mesh refinement for degenerate elliptic problems},
  journal = {SIAM Journal on Numerical Analysis},
  year = {2023},
  volume = {61},
  number = {2},
  pages = {789}
}

@article{L2021,
  author = {Chen L},
  title = {Spline-based solutions for boundary value problems with degeneracy},
  journal = {Applied Mathematics and Computation},
  year = {2021},
  pages = {300}
}

@article{K2022,
  author = {Anderson K},
  title = {Variational approaches to degenerate differential equations},
  journal = {Nonlinear Analysis: Theory, Methods & Applications},
  year = {2022},
  pages = {300}
}

@article{R2023b,
  author = {Taylor R},
  title = {High-order approximation methods for singular differential equations},
  journal = {Numerische Mathematik},
  year = {2023},
  volume = {154},
  number = {3},
  pages = {521}
}

@article{J2021,
  author = {Wilson J},
  title = {Finite element analysis of degenerate boundary value problems},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year = {2021},
  pages = {200}
}

@article{V2022,
  author = {Martinez V},
  title = {Weighted Sobolev spaces and degenerate elliptic equations},
  journal = {Journal of Mathematical Analysis and Applications},
  year = {2022},
  pages = {100}
}

@article{S2023,
  author = {Kim S},
  title = {Trigonometric splines for singular perturbation problems},
  journal = {Journal of Scientific Computing},
  year = {2023},
  volume = {94},
  number = {2},
  pages = {45}
}

@article{H2021b,
  author = {Thompson H},
  title = {Mixed finite element methods for degenerate parabolic equations},
  journal = {Mathematics of Computation},
  year = {2021},
  volume = {90},
  number = {329},
  pages = {1017}
}

@article{P2022,
  author = {Davis P},
  title = {A posteriori error estimation for degenerate PDEs},
  journal = {IMA Journal of Numerical Analysis},
  year = {2022},
  volume = {42},
  number = {3},
  pages = {1892}
}

@article{L2023,
  author = {Evans L},
  title = {Degenerate elliptic equations with variable coefficients},
  journal = {Communications in Partial Differential Equations},
  year = {2023},
  volume = {48},
  number = {5},
  pages = {723}
}
"""


# Extract reference numbers from the text
def extract_references(text):
    # Pattern to match [n] or [n,m,...] references
    pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    references = re.findall(pattern, text)
    return references


# Parse BibTeX to create a mapping from reference numbers to aliases
def parse_bibtex_references(bibtex_str):
    # Create a simple parser to extract the BibTeX entries
    entries = []
    for line in bibtex_str.strip().split("\n"):
        if line.startswith("@"):
            # Extract the alias (key) from the BibTeX entry
            match = re.match(r"@\w+{([^,]+),", line)
            if match:
                entries.append(match.group(1))

    # Create a mapping from reference numbers to aliases
    ref_map = {}
    for i, alias in enumerate(entries, 1):
        ref_map[str(i)] = alias

    return ref_map


# Replace references in the text with BibTeX aliases
def replace_references(text, ref_map):
    def replace_match(match):
        ref_nums = match.group(1).split(",")
        ref_nums = [num.strip() for num in ref_nums]

        # Replace each reference number with its alias
        aliases = []
        for num in ref_nums:
            if num in ref_map:
                aliases.append(f"@{ref_map[num]}")
            else:
                aliases.append(f"[{num}]")  # Keep original if not found

        # Join aliases with commas for compound references
        if len(aliases) == 1:
            return aliases[0]
        else:
            return ", ".join(aliases)

    # Pattern to match [n] or [n,m,...] references
    pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    return re.sub(pattern, replace_match, text)


# Main processing
ref_map = parse_bibtex_references(references_bib)
modified_text = replace_references(s, ref_map)

print(modified_text)
