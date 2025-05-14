### Note
[X]dal dizionario _labelcounts_ in _ensembling.py_, quando le label sono tied viene scelta una a caso tra le due, la prima che compare. C'è un seed. 



[X]*Annotatori senza "income":*
    9226
    {'Gender': ['prefer_not_to_say']}
    3797
    {'Education': ['educ-high'], 'Gender': ['female'], 'Ideology': ['liberal'], 'Age': ['GenY']}
    6749
    {'Education': ['educ-high'], 'Gender': ['prefer_not_to_say'], 'Ideology': ['neutral'], 'Age': ['GenY']}


[X] Cross entropy non va bene con lamp, dove il modello può aver dato due label diverse con lo stesso prompt dato che ha visto esempi diversi. 

[X] Reference to Jensen-Shannon Divergence --> https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/77187/UMA_Alexandra_170576754_EECS_PhD_final.pdf?sequence=1 (p.48)


[]line 694 Data.py --> cambiare "educ-high" a "master"