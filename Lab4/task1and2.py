from nltk import CFG, ChartParser

grammar = CFG.fromstring("""
S -> NP VP | NP VP NP
NP -> Det N | Det Adj N | N | Adj N | NP PP | NP Conj NP| VG N | NP Conj NP |PP NP | NP V NP
VP -> V | V NP | V PP | Aux V | VP Adj | V Adv | Aux VG
PP -> P NP | Adv P
Det -> 'the'
N -> 'planes' | 'parents' | 'bride' | 'groom' | 'act'
Adj -> 'flying' | 'dangerous'
V -> 'be' | 'loves'
VG -> 'flying'
Aux -> 'can' | 'were'
P -> 'of' | 'than'
Conj -> 'and'
Adv -> 'more'
""")

parser = ChartParser(grammar)

sentences = [
    "Flying planes can be dangerous".split(),
    "The parents of the bride and the groom were flying".split(),
    "The groom loves dangerous planes more than the bride".split()
]

for sent in sentences:
    sent = [word.lower() for word in sent]
    print(f"\nSentence: {' '.join(sent)}")
    for tree in parser.parse(sent):
        print(tree)
        tree.pretty_print()
