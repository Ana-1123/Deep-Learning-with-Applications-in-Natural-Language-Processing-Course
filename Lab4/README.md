# Lab 4 - Parsing and Grammar Analysis

## Task 1 & 2: Context-Free Grammar Implementation
### Implementation Details
- Created a context-free grammar (CFG) using NLTK's CFG class
- Implemented parsing using NLTK's ChartParser

### Libraries Used
- NLTK (Natural Language Toolkit)

### Challenges
- Defining ambiguous grammar rules for sentences like "Flying planes can be dangerous"
- Handling multiple valid parse trees for the same sentence
- Creating comprehensive rules to cover all sentence variations
- Dealing with part-of-speech ambiguity

## Task 3: Dependency Parsing
### Implementation Details
- Used spaCy for dependency parsing
- Analyzed sentence structure through:
  - Token relationships
  - Head-dependent relationships
  - Dependency types

### Libraries Used
- spaCy
- en_core_web_sm model

## Bonus Task: CFG to CNF Converter
### Implementation Details
- Implemented a converter from Context-Free Grammar to Chomsky Normal Form
- Developed steps for:
  - Eliminating ε-rules
  - Removing unit rules
  - Converting mixed rules
  - Breaking long rules

### Libraries Used
- collections (defaultdict, deque)
- re (regular expressions)

### Challenges
- Maintaining probability distributions during rule conversion
- Handling epsilon rules correctly
- Generating unique non-terminal symbols
- Preserving grammar semantics during transformation