from collections import defaultdict, deque
import re

class CFGtoCNFConverter:
    def __init__(self):
        self.terminals = set()
        self.non_terminals = set()
        self.start_symbol = None
        self.rules = defaultdict(list)
        self.new_non_terminals_count = 0
        
    def _get_new_non_terminal(self, base="X"):
        """Generate new non-terminal symbols"""
        self.new_non_terminals_count += 1
        new_symbol = f"{base}_{self.new_non_terminals_count}"
        self.non_terminals.add(new_symbol)
        return new_symbol
    
    def read_grammar(self, grammar_input):
        """Read grammar from string or file"""
        if isinstance(grammar_input, str):
            lines = grammar_input.strip().split('\n')
        else:
            lines = grammar_input
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse rule: A -> B c [0.5]
            match = re.match(r'(\w+)\s*->\s*(.+?)(?:\s*\[([\d.]+)\])?$', line)
            if not match:
                continue
                
            lhs, rhs, prob_str = match.groups()
            probability = float(prob_str) if prob_str else 1.0
            
            # Set start symbol if not set
            if self.start_symbol is None:
                self.start_symbol = lhs
                
            self.non_terminals.add(lhs)
            
            # Split RHS and identify terminals/non-terminals
            rhs_symbols = []
            for symbol in rhs.split():
                if symbol == 'ε':
                    rhs_symbols.append('')
                elif symbol.islower() or not symbol.isalpha():
                    self.terminals.add(symbol)
                    rhs_symbols.append(symbol)
                else:
                    self.non_terminals.add(symbol)
                    rhs_symbols.append(symbol)
            
            self.rules[lhs].append((rhs_symbols, probability))
    
    def _normalize_probabilities(self):
        """Ensure probabilities for each LHS sum to 1"""
        for lhs in self.rules:
            total_prob = sum(prob for _, prob in self.rules[lhs])
            if total_prob > 0:
                for i in range(len(self.rules[lhs])):
                    rhs, prob = self.rules[lhs][i]
                    self.rules[lhs][i] = (rhs, prob / total_prob)
    
    def step1_eliminate_epsilon(self):
        """Step 1: Eliminate ε-rules (except possibly S -> ε)"""
        # Find all nullable non-terminals
        nullable = set()
        changed = True
        
        while changed:
            changed = False
            for lhs in self.rules:
                for rhs, prob in self.rules[lhs]:
                    # Check if all symbols in RHS are nullable or empty
                    if all(sym == '' or sym in nullable for sym in rhs):
                        if lhs not in nullable:
                            nullable.add(lhs)
                            changed = True
        
        # Generate new rules without ε
        new_rules = defaultdict(list)
        
        for lhs in self.rules:
            for rhs, prob in self.rules[lhs]:
                # For each possible combination of nullable symbols
                indices = [i for i, sym in enumerate(rhs) if sym in nullable]
                
                # Generate all subsets of nullable positions to remove
                from itertools import combinations, chain
                all_combinations = chain.from_iterable(
                    combinations(indices, r) for r in range(len(indices) + 1)
                )
                
                for remove_set in all_combinations:
                    new_rhs = [sym for i, sym in enumerate(rhs) 
                              if i not in remove_set and sym != '']
                    
                    if new_rhs or (not rhs and not remove_set):
                        new_prob = prob 
                        new_rules[lhs].append((new_rhs, new_prob))
        
        self.rules = new_rules
        self._normalize_probabilities()
    
    def step2_eliminate_unit_rules(self):
        """Step 2: Eliminate unit rules (A -> B)"""
        new_rules = defaultdict(list)
        
        for lhs in self.rules:
            # Find all non-terminals reachable through unit productions
            reachable = set()
            queue = deque([lhs])
            
            while queue:
                current = queue.popleft()
                for rhs, prob in self.rules[current]:
                    if len(rhs) == 1 and rhs[0] in self.non_terminals:
                        if rhs[0] not in reachable:
                            reachable.add(rhs[0])
                            queue.append(rhs[0])
            
            # Add all non-unit productions from reachable symbols
            for symbol in reachable.union({lhs}):
                for rhs, prob in self.rules[symbol]:
                    if not (len(rhs) == 1 and rhs[0] in self.non_terminals):
                        # Distribute probability 
                        new_prob = prob / len(reachable.union({lhs})) if reachable else prob
                        new_rules[lhs].append((rhs, new_prob))
        
        self.rules = new_rules
        self._normalize_probabilities()
    
    def step3_eliminate_mixed_rules(self):
        """Step 3: Eliminate rules with mixed terminals and non-terminals"""
        terminal_rules = {}
        new_rules = defaultdict(list)
        
        # First pass: identify which terminals appear in mixed rules
        terminals_needed = set()
        for lhs in self.rules:
            for rhs, prob in self.rules[lhs]:
                # Check if this is a mixed rule (has both terminals and non-terminals)
                has_terminals = any(sym in self.terminals for sym in rhs)
                has_non_terminals = any(sym not in self.terminals and sym != '' for sym in rhs)
                
                if has_terminals and has_non_terminals:
                    # This is a mixed rule - mark all terminals in it as needed
                    for symbol in rhs:
                        if symbol in self.terminals:
                            terminals_needed.add(symbol)
        
        # Create rules for terminals that actually appear in mixed rules
        for terminal in terminals_needed:
            new_nt = self._get_new_non_terminal()
            terminal_rules[terminal] = new_nt
            new_rules[new_nt].append(([terminal], 1.0))
        
        # Convert existing rules
        for lhs in self.rules:
            for rhs, prob in self.rules[lhs]:
                # Check if this rule has both terminals and non-terminals
                has_terminals = any(sym in self.terminals for sym in rhs)
                has_non_terminals = any(sym not in self.terminals and sym != '' for sym in rhs)
                
                # Case 1: Single terminal or all non-terminals - keep as is
                if (len(rhs) == 1 and rhs[0] in self.terminals) or not has_terminals:
                    new_rules[lhs].append((rhs, prob))
                
                # Case 2: Mixed terminals and non-terminals
                elif has_terminals and has_non_terminals:
                    new_rhs = []
                    for symbol in rhs:
                        if symbol in self.terminals:
                            new_rhs.append(terminal_rules[symbol])
                        else:
                            new_rhs.append(symbol)
                    new_rules[lhs].append((new_rhs, prob))
                
                # Case 3: All terminals (will be handled in step 4)
                else:
                    new_rules[lhs].append((rhs, prob))
        
        self.rules = new_rules
        
    def step4_eliminate_long_rules(self):
        """Step 4: Break rules with more than 2 symbols on RHS"""
        new_rules = defaultdict(list)
        
        for lhs in self.rules:
            for rhs, prob in self.rules[lhs]:
                if len(rhs) <= 2:
                    new_rules[lhs].append((rhs, prob))
                else:
                    # Break A -> BCD into A -> B X_1, X_1 -> CD
                    current_lhs = lhs
                    remaining_rhs = rhs
                    
                    while len(remaining_rhs) > 2:
                        new_nt = self._get_new_non_terminal()
                        first_one = remaining_rhs[:1]
                        new_rules[current_lhs].append((first_one + [new_nt], prob))
                        current_lhs = new_nt
                        remaining_rhs = remaining_rhs[1:]
        
        self.rules = new_rules
    
    def convert_to_cnf(self):
        print("Original Grammar:")
        self.print_grammar()
        
        print("\nStep 1: Eliminating ε-rules...")
        self.step1_eliminate_epsilon()
        self.print_grammar()
        
        print("\nStep 2: Eliminating unit rules...")
        self.step2_eliminate_unit_rules()
        self.print_grammar()
        
        print("\nStep 3: Eliminating mixed rules...")
        self.step3_eliminate_mixed_rules()
        self.print_grammar()
        
        print("\nStep 4: Eliminating long rules...")
        self.step4_eliminate_long_rules()
        self.print_grammar()
        
        print(f"\nConversion complete. Generated {self.new_non_terminals_count} new non-terminals")
    
    def print_grammar(self):
        """Print the current grammar rules"""
        for lhs in sorted(self.rules.keys()):
            for rhs, prob in self.rules[lhs]:
                rhs_str = ' '.join(rhs) if rhs else 'ε'
                print(f"{lhs} -> {rhs_str} [{prob:.3f}]")

if __name__ == "__main__":
    # Example probabilistic CFG
    example_grammar = """
    S -> NP VP [1.0]
    NP -> Det N [0.5]
    NP -> N [0.3]
    NP -> Det Adj N [0.2]        
    VP -> V NP [0.6]
    VP -> V [0.3]
    VP -> 'quickly' V [0.1]      
    Det -> 'the' [1.0]
    N -> 'cat' [0.5]
    N -> 'dog' [0.5]
    Adj -> 'big' [1.0]
    V -> 'chases' [1.0]
    """
    
    converter = CFGtoCNFConverter()
    converter.read_grammar(example_grammar)
    converter.convert_to_cnf()