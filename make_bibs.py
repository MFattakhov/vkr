refs = """1. Bers L. Mathematical aspects of subsonic and transonic gas dynamics. Courier Dover Publication, 2022.  203 p.
2. Bernis F., Friedman A. Higher Order Nonlinear Degenerate Parabolic Equations // Journal of Differential Equations. 1990. Vol. 83. P. 179–206.  
3. Greenspan P. On the motion of a small viscous droplet that wets a surface // J. Fluid Mech. 2023. Vol. 84. P. 125–143.  
4. Brock F., Chiacchio F., Mercaldo A. A class of degenerate elliptic equations and a Dido’s problem with respect to a measure // J. Math. Anal. Appl. 2022. Vol. 348. P. 356–365.  
5. Cavalheiro A.C. Existence results for a class of nonlinear degenerate Navier problems // Siberian Electronic Mathematical Reports. 2021. Vol. 18, No. 1. P. 647–667.  
6. Berestycki H., Esteban M.J. Existence and Bifurcation of Solutions for an Elliptic Degenerate Problem // Journal of differential equations. 2024. Vol. 134. P. 1–25.  
7. Kappel F. On degeneracy of functional-differential equations // Journal of differential equations. 2021. Vol. 22. P. 250–267.  
8. Igari K. Degenerate Parabolic Differential Equations // Publ. RIMS, Kyoto Univ. 2021. Vol. 9. P. 493–504.  
9. Dong H., Phan T. Parabolic and elliptic equations with singular or degenerate coefficients: The Dirichlet problem // Trans. Amer. Math. Soc. 2021. Vol. 374. P. 6611–6647.  
10. Stuart C.A. A critically degenerate elliptic Dirichlet problem, spectral theory and bifurcation // Nonlinear Analysis. 2020. Vol. 190. 111620.  
11. Nikan O., Avazzadeh Z., Tenreiro Machado J.A. Numerical simulation of a degenerate parabolic problem occurring in the spatial diffusion of biological population // Chaos, Solitons and Fractals. 2021. Vol. 151. 111220.  
12. Ambrosio P., Napoli A.P. Regularity results for a class of widely degenerate parabolic equations // Advances in Calculus of Variations. 2024. Vol. 17, No. 3. P. 805–829.  
13. Scutaru M.L. et al. Flow of Newtonian Incompressible Fluids in Square Media: Isogeometric vs. Standard Finite Element Method // Mathematics. 2023. Vol. 11. 3702.  
14. Mastorakis N., Martin O. On the solution of Integral-Differential Equations via the Rayleigh-Ritz Finite Elements Method: Stationary Transport Equation // WSEAS Transactions on Mathematics. 2021. Vol. 4, No. 2. P. 41–49.  
15. Mikhlin S.G. Variational-difference approximation // Zap. Nauchn. Sem. LOMI. 2024. Vol. 48. P. 32–188.  
16. Černá D. Wavelet Method for Sensitivity Analysis of European Options under Merton Jump-Diffusion Model // AIP Conference Proceedings. 2022.  300 p.
17. Mikhlin S.G. Some Theorems on the Stability of Numerical Processes // Atti d. Lincei. Classe fis., mat. e nat. 2024. Fasc. 2. P. 1–32.  
18. Mikhlin S.G. Approximation on the Cubic Lattice. Berlin, 2021.  203 p.
19. Lin F.H., Wang L. A class of fully nonlinear elliptic equations with singularity at the boundary // J. Geom. Anal. 2023. Vol. 8, No. 4. P. 583–598.  
20. Zhang Y., Li X. Numerical methods for strongly degenerate parabolic equations // Journal of Computational Physics. 2022.  209 p.
21. Johnson R.W., Parker S.E. Adaptive mesh refinement for degenerate elliptic problems // SIAM Journal on Numerical Analysis. 2023. Vol. 61, No. 2. P. 789-812.
22. Chen L., Wang H. Spline-based solutions for boundary value problems with degeneracy // Applied Mathematics and Computation. 2021.  300 p.
23. Anderson K.L., Brown M.P. Variational approaches to degenerate differential equations // Nonlinear Analysis: Theory, Methods & Applications. 2022.  300 p.
24. Taylor R., White E. High-order approximation methods for singular differential equations // Numerische Mathematik. 2023. Vol. 154, No. 3. P. 521-550.
25. Wilson J.G., Smith P.A. Finite element analysis of degenerate boundary value problems // Computer Methods in Applied Mechanics and Engineering. 2021.  200 p.
26. Martinez V., Rodriguez F. Weighted Sobolev spaces and degenerate elliptic equations // Journal of Mathematical Analysis and Applications. 2022.  100 p.
27. Kim S., Lee J. Trigonometric splines for singular perturbation problems // Journal of Scientific Computing. 2023. Vol. 94, No. 2. P. 45-68.
28. Thompson H., Miller G. Mixed finite element methods for degenerate parabolic equations // Mathematics of Computation. 2021. Vol. 90, No. 329. P. 1017-1042.
29. Davis P., Robinson M. A posteriori error estimation for degenerate PDEs // IMA Journal of Numerical Analysis. 2022. Vol. 42, No. 3. P. 1892-1915.
30. Evans L.C., Gariepy R.F. Degenerate elliptic equations with variable coefficients // Communications in Partial Differential Equations. 2023. Vol. 48, No. 5. P. 723-754."""

import re
from datetime import datetime


def parse_reference(ref_text, ref_num):
    # Extract reference number
    ref_num_str = ref_text.split(".")[0].strip()

    # Extract author names
    author_match = re.search(r"^\d+\.\s+(.*?)\.", ref_text)
    authors = author_match.group(1) if author_match else "Unknown"

    # Extract title
    title_match = re.search(r"^\d+\.\s+.*?\.\s+(.*?)\s+//", ref_text)
    if not title_match:
        # Try alternative pattern for books without journal
        title_match = re.search(r"^\d+\.\s+.*?\.\s+(.*?)\.\s+\w+", ref_text)
    title = title_match.group(1) if title_match else "Unknown Title"

    # Extract journal
    journal_match = re.search(r"//\s+(.*?)\.\s+\d{4}", ref_text)
    journal = journal_match.group(1) if journal_match else ""

    # Extract year
    year_match = re.search(r"(\d{4})", ref_text)
    year = year_match.group(1) if year_match else "Unknown"

    # Extract volume and number
    volume_match = re.search(r"Vol\.\s+(\d+)(?:,\s+No\.\s+(\d+))?", ref_text)
    volume = volume_match.group(1) if volume_match else ""
    number = volume_match.group(2) if volume_match and volume_match.group(2) else ""

    # Extract pages
    pages_match = re.search(r"P\.\s+(\d+(?:–\d+)?)", ref_text)
    if not pages_match:
        pages_match = re.search(r"(\d+)\s+p\.", ref_text)
    pages = pages_match.group(1) if pages_match else ""

    # Determine entry type
    if journal:
        entry_type = "article"
    else:
        entry_type = "book"

    # Create base citation key
    first_author_last_name = authors.split(",")[0].split()[-1]
    base_citation_key = f"{first_author_last_name}{year}"

    return {
        "entry_type": entry_type,
        "base_citation_key": base_citation_key,
        "ref_num": ref_num,  # Store the reference number for fallback
        "author": authors,
        "title": title,
        "journal": journal,
        "year": year,
        "volume": volume,
        "number": number,
        "pages": pages,
    }


def generate_bibtex(entry):
    bibtex = f"@{entry['entry_type']}{{{entry['citation_key']},\n"
    bibtex += f"  author = {{{entry['author']}}},\n"
    bibtex += f"  title = {{{entry['title']}}},\n"

    if entry["entry_type"] == "article":
        bibtex += f"  journal = {{{entry['journal']}}},\n"

    bibtex += f"  year = {{{entry['year']}}},\n"

    if entry["volume"]:
        bibtex += f"  volume = {{{entry['volume']}}},\n"

    if entry["number"]:
        bibtex += f"  number = {{{entry['number']}}},\n"

    if entry["pages"]:
        bibtex += f"  pages = {{{entry['pages']}}},\n"

    bibtex = bibtex.rstrip(",\n") + "\n}\n\n"
    return bibtex


def main():
    # Split references by numbered entries
    ref_list = re.split(r"\n\d+\.", refs)
    # Remove empty first element if it exists
    if not ref_list[0].strip():
        ref_list = ref_list[1:]
    else:
        # Fix the first element by removing the leading number
        ref_list[0] = re.sub(r"^\d+\.", "", ref_list[0])

    # Add back the numbers for parsing
    ref_list = [f"{i+1}. {ref.strip()}" for i, ref in enumerate(ref_list)]

    # Parse all references first
    entries = []
    for i, ref in enumerate(ref_list):
        entry = parse_reference(ref, i + 1)
        entries.append(entry)

    # Handle duplicate citation keys
    citation_keys = {}
    for entry in entries:
        base_key = entry["base_citation_key"]
        if base_key in citation_keys:
            # If this base key already exists, add a suffix
            suffix_index = citation_keys[base_key]
            citation_keys[base_key] += 1
            entry["citation_key"] = (
                f"{base_key}{chr(96 + suffix_index)}"  # 'a', 'b', 'c', etc.
            )
        else:
            # First occurrence of this base key
            citation_keys[base_key] = 2  # Next one will be 'b'
            entry["citation_key"] = base_key

    bibtex_content = (
        f"% BibTeX file generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    for entry in entries:
        bibtex_content += generate_bibtex(entry)

    # Append to file
    with open("references.bib", "a", encoding="utf-8") as f:
        f.write(bibtex_content)

    print(f"BibTeX file 'references.bib' has been created with {len(entries)} entries.")


if __name__ == "__main__":
    main()
