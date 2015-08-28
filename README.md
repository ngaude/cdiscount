#[Catégorisation de produits pour le e-commerce](https://www.datascience.net/fr/challenge/20/details)

*cdiscount opère une offre “marketplace” qui référence des produits en provenance de commerçants partenaires. Pour assurer une visibilité maximale de ces produits sur son site,
cdiscount doit assurer une catégorisation homogène de ces produits aux origines variées.*

Solution structurée autour d'une approche bagging / staging via un Ensemble de pyramides de Régressions Logistiques : «staged-logistic-regressions ensemble»

Note méthodologique complête: 
[note_methodologique.pdf](https://github.com/ngaude/cdiscount/note_methodologique.pdf)

Reproduire les résultats d'un ensemble à 5 pyramides (67,4%):
[pipeline.sh](https://github.com/ngaude/cdiscount/pipeline.sh)
