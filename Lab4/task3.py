import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

sentences = [
    "Flying planes can be dangerous.",
    "The parents of the bride and the groom were flying.",
    "The groom loves dangerous planes more than the bride."
]

# Parse each sentence
for sent in sentences:
    doc = nlp(sent)
    print(f"\nSentence: {sent}")
    print("-" * 60)
    
    # Print dependency information
    print(f"{'Token':<12} {'Head':<12} {'Dep':<10} {'Children'}")
    for token in doc:
        children = [child.text for child in token.children]
        print(f"{token.text:<12} {token.head.text:<12} {token.dep_:<10} {children}")
