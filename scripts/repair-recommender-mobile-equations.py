from pathlib import Path
import re

path = Path('src/learn/data/topics/recommender-systems-collaborative-filtering-matrix-factorization.jsx')
source = path.read_text(encoding='utf-8')
pattern = re.compile(r'<MathBlock>\{String\.raw`(.*?)`\}</MathBlock>', re.S)
matches = list(pattern.finditer(source))
assert len(matches) == 14
repairs = {
    1: r'''\begin{aligned}
      C_{ij}&=\sum_{v\in S_{ij}}a_{vi}a_{vj},\\
      N_i&=\sqrt{\sum_{v\in S_{ij}}a_{vi}^2},\\
      N_j&=\sqrt{\sum_{v\in S_{ij}}a_{vj}^2},\\
      \operatorname{cos}(i,j)&=C_{ij}/(N_iN_j).
    \end{aligned}''',
    4: r'''\begin{aligned}
      \ell_{ui}&=\tfrac12 e_{ui}^2+\tfrac\lambda2(\|p_u\|^2+\|q_i\|^2)\\
      &\quad+\tfrac\lambda2(b_u^2+b_i^2).
    \end{aligned}''',
    8: r'''\begin{aligned}
      E&=\sum_{u,i}c_{ui}(z_{ui}-p_u^\top q_i)^2,\\
      J&=E+\lambda\sum_u\|p_u\|^2\\
      &\quad+\lambda\sum_i\|q_i\|^2.
    \end{aligned}''',
    9: r'''\begin{aligned}
      Q^\top C_uQ&=Q^\top Q\\
      &\quad+\sum_{i:n_{ui}>0}(c_{ui}-1)q_iq_i^\top,\\
      Q^\top C_uz_u&=\sum_{i:n_{ui}>0}c_{ui}q_i.
    \end{aligned}''',
    10: r'''\begin{aligned}
      \ell_{uij}&=\log(1+e^{-\Delta})\\
      &\quad+\tfrac\lambda2(\|p_u\|^2+\|q_i\|^2)\\
      &\quad+\tfrac\lambda2\|q_j\|^2.
    \end{aligned}''',
}
for index in sorted(repairs, reverse=True):
    match = matches[index]
    source = source[:match.start(1)] + repairs[index] + source[match.end(1):]
source = source.replace('For raw cosine, aᵥᵢ=rᵥᵢ.', 'All three sums use the same shared set Sᵢⱼ, so the lengths Nᵢ and Nⱼ belong to this particular pair. For raw cosine, aᵥᵢ=rᵥᵢ.')
source = source.replace('This sum includes <strong>all user–item pairs</strong>.', 'E is the data-error sum, and J adds the penalties. This sum includes <strong>all user–item pairs</strong>.')
path.write_text(source, encoding='utf-8')
print('Reflowed five equivalent equations; retained every term and defined local intermediates.')
