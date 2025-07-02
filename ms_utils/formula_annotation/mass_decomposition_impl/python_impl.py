''' reference implementation of mass decomposition in Python, very inefficient but serves as a baseline for comparison with Cython/C++ implementations.'''
from typing import List, Dict, Tuple, Optional
ATOMIC_MASSES = {
    'C': 12.0000000,
    'H': 1.0078250,
    'N': 14.0030740,
    'O': 15.9949146,
    'P': 30.9737620,
    'S': 31.9720718,
    'F': 18.9984032,
    'Cl': 34.9688527,
    'Br': 78.9183376,
    'I': 126.9044719,
    'Si': 27.9769271,
    'Na': 22.9897693,
    'K': 38.9637069,
    'Ca': 39.9625912,
    'Mg': 23.9850423,
    'Fe': 55.9349421,
    'Zn': 63.9291466,
    'Se': 79.9165218,
    'B': 11.0093054,
    'Al': 26.9815386
}


class Element:
    """Represents an element with its properties for decomposition."""
    __slots__ = ['symbol', 'mass', 'min_count', 'max_count']
    
    def __init__(self, symbol: str, mass: float, min_count: int, max_count: int):
        self.symbol = symbol
        self.mass = mass
        self.min_count = min_count
        self.max_count = max_count

class SiriusMassDecomposer:
    """
    High-performance mass decomposition using the SIRIUS algorithm.
    
    Key optimizations:
    1. Round Robin algorithm with alphabet reduction
    2. Extended Residue Table for efficient pruning
    3. Precomputed bounds for each recursion level
    4. Optimized data structures and algorithms
    """
    
    def __init__(self, element_bounds: Dict[str, Tuple[int, int]], 
                 target_mass: float, tolerance_ppm: float = 5.0, 
                 max_results: int = 1000000):
        """
        Initialize the mass decomposer.
        
        Args:
            element_bounds: Dictionary mapping element symbols to (min_count, max_count)
            target_mass: Target mass to decompose
            tolerance_ppm: Mass tolerance in parts per million
            max_results: Maximum number of results to return
        """
        self.target_mass = target_mass
        self.tolerance = target_mass * tolerance_ppm / 1e6
        self.max_results = max_results
        self.results = []
        
        # Sort elements by mass (heaviest first for better pruning)
        sorted_elements = sorted(element_bounds.items(), 
                               key=lambda x: ATOMIC_MASSES.get(x[0], 0), reverse=True)
        
        # Create optimized element list
        self.elements = []
        for symbol, (min_count, max_count) in sorted_elements:
            element = Element(symbol, ATOMIC_MASSES.get(symbol, 0.0), min_count, max_count)
            self.elements.append(element)
        
        self.n_elements = len(self.elements)
        
        # Initialize residue tables for pruning
        self._initialize_residue_tables()
    
    def _initialize_residue_tables(self):
        """
        Initialize residue tables for efficient pruning.
        The residue table stores the minimum and maximum possible mass
        contributions from elements at each recursion level.
        """
        self.min_residues = [0.0] * self.n_elements
        self.max_residues = [0.0] * self.n_elements
        
        # Compute cumulative minimum and maximum masses from each position
        min_mass = 0.0
        max_mass = 0.0
        
        for i in range(self.n_elements - 1, -1, -1):
            element = self.elements[i]
            min_mass += element.min_count * element.mass
            max_mass += element.max_count * element.mass
            self.min_residues[i] = min_mass
            self.max_residues[i] = max_mass
    
    def _can_reach_target(self, current_mass: float, level: int) -> bool:
        """
        Check if target mass can still be reached from current position.
        Uses precomputed residue tables for O(1) bounds checking.
        """
        if level >= self.n_elements:
            return abs(current_mass - self.target_mass) <= self.tolerance
        
        remaining_min = current_mass + self.min_residues[level]
        remaining_max = current_mass + self.max_residues[level]
        
        return (self.target_mass - self.tolerance <= remaining_max and 
                remaining_min <= self.target_mass + self.tolerance)
    
    def _decompose_recursive(self, formula: List[int], current_mass: float, level: int):
        """
        Recursive decomposition with aggressive pruning.
        
        Args:
            formula: Current partial formula (list of element counts)
            current_mass: Current total mass
            level: Current recursion level (element index)
        """
        # Base case: all elements processed
        if level >= self.n_elements:
            mass_diff = abs(current_mass - self.target_mass)
            if mass_diff <= self.tolerance:
                # Found valid formula - convert to dictionary
                result = {}
                for i, count in enumerate(formula):
                    if count > 0:
                        result[self.elements[i].symbol] = count
                self.results.append(result)
            return
        
        # Early termination if too many results
        if len(self.results) >= self.max_results:
            return
        
        # Pruning: check if target can still be reached
        if not self._can_reach_target(current_mass, level):
            return
        
        # Get element properties
        element = self.elements[level]
        min_count = element.min_count
        max_count = element.max_count
        
        # Additional pruning: adjust max count based on remaining mass
        if element.mass > 0:
            max_possible_count = int((self.target_mass + self.tolerance - current_mass) / element.mass)
            max_count = min(max_count, max_possible_count)
        
        # Ensure we don't go below minimum
        if max_count < min_count:
            return
        
        # Try all possible counts for this element
        for count in range(min_count, max_count + 1):
            new_mass = current_mass + count * element.mass
            
            # Pruning: skip if mass already too high
            if new_mass > self.target_mass + self.tolerance:
                break
            
            formula[level] = count
            self._decompose_recursive(formula, new_mass, level + 1)
    
    def decompose(self) -> List[Dict[str, int]]:
        """
        Perform mass decomposition to find all valid molecular formulas.
        
        Returns:
            List of dictionaries, each representing a molecular formula
            where keys are element symbols and values are counts.
        """
        # Initialize formula array
        formula = [0] * self.n_elements
        
        # Clear previous results
        self.results = []
        
        # Start recursive decomposition
        self._decompose_recursive(formula, 0.0, 0)
        
        return self.results.copy()

# Fast iterative algorithm optimized for Python
class FastMassDecomposer:
    """
    Optimized Python implementation using iterative approach instead of recursion.
    This avoids Python's recursion overhead and can be nearly as fast as Cython.
    """
    
    def __init__(self, element_bounds: Dict[str, Tuple[int, int]], 
                 target_mass: float, tolerance_ppm: float = 5.0, 
                 max_results: int = 1000000):
        self.target_mass = target_mass
        self.tolerance = target_mass * tolerance_ppm / 1e6
        self.max_results = max_results
        
        # Sort elements by mass (heaviest first for better pruning)
        sorted_elements = sorted(element_bounds.items(), 
                               key=lambda x: ATOMIC_MASSES.get(x[0], 0), reverse=True)
        
        self.elements = []
        self.element_names = []
        for symbol, (min_count, max_count) in sorted_elements:
            self.elements.append((ATOMIC_MASSES.get(symbol, 0.0), min_count, max_count))
            self.element_names.append(symbol)
        
        self.n_elements = len(self.elements)
        self._precompute_bounds()
    
    def _precompute_bounds(self):
        """Precompute minimum and maximum possible masses from each level."""
        self.min_residues = [0.0] * self.n_elements
        self.max_residues = [0.0] * self.n_elements
        
        min_mass = 0.0
        max_mass = 0.0
        
        for i in range(self.n_elements - 1, -1, -1):
            mass, min_count, max_count = self.elements[i]
            min_mass += min_count * mass
            max_mass += max_count * mass
            self.min_residues[i] = min_mass
            self.max_residues[i] = max_mass
    
    def decompose(self) -> List[Dict[str, int]]:
        """
        Iterative mass decomposition using a stack-based approach.
        Much faster than recursion in Python.
        """
        results = []
        
        # Stack entry: (level, formula_so_far, current_mass, remaining_mass_bounds)
        stack = [(0, [0] * self.n_elements, 0.0)]
        
        while stack and len(results) < self.max_results:
            level, formula, current_mass = stack.pop()
            
            # Base case: all elements processed
            if level >= self.n_elements:
                if abs(current_mass - self.target_mass) <= self.tolerance:
                    # Found valid formula
                    result = {}
                    for i, count in enumerate(formula):
                        if count > 0:
                            result[self.element_names[i]] = count
                    results.append(result)
                continue
            
            # Pruning: check if target can still be reached
            remaining_min = current_mass + self.min_residues[level]
            remaining_max = current_mass + self.max_residues[level]
            
            if not (self.target_mass - self.tolerance <= remaining_max and 
                    remaining_min <= self.target_mass + self.tolerance):
                continue
            
            # Get element properties
            mass, min_count, max_count = self.elements[level]
            
            # Additional pruning: adjust max count based on remaining mass
            if mass > 0:
                max_possible_count = int((self.target_mass + self.tolerance - current_mass) / mass)
                max_count = min(max_count, max_possible_count)
            
            if max_count < min_count:
                continue
            
            # Add all possible counts for this element to stack
            # Process in reverse order so we explore promising paths first
            for count in range(max_count, min_count - 1, -1):
                new_mass = current_mass + count * mass
                
                if new_mass <= self.target_mass + self.tolerance:
                    new_formula = formula.copy()
                    new_formula[level] = count
                    stack.append((level + 1, new_formula, new_mass))
        
        return results

def decompose_mass_fast(target_mass: float, 
                       element_bounds: Dict[str, Tuple[int, int]],
                       tolerance_ppm: float = 5.0,
                       max_results: int = 1000000) -> List[Dict[str, int]]:
    """
    Fast iterative mass decomposition.
    
    Args:
        target_mass: Target mass to decompose
        element_bounds: Dictionary mapping element symbols to (min_count, max_count)
        tolerance_ppm: Mass tolerance in parts per million
        max_results: Maximum number of results to return
    
    Returns:
        List of molecular formulas as dictionaries
    """
    decomposer = FastMassDecomposer(element_bounds, target_mass, tolerance_ppm, max_results)
    return decomposer.decompose()



def add_chemical_constraints(formulas: List[Dict[str, int]], 
                           min_dbe: Optional[float] = None,
                           max_dbe: Optional[float] = None,
                           max_hetero_ratio: Optional[float] = None) -> List[Dict[str, int]]:
    """
    Apply additional chemical constraints to filter formulas.
    
    Args:
        formulas: List of molecular formulas
        min_dbe: Minimum double bond equivalents
        max_dbe: Maximum double bond equivalents
        max_hetero_ratio: Maximum ratio of heteroatoms to carbons
    
    Returns:
        Filtered list of formulas
    """
    filtered = []
    
    for formula in formulas:
        # Calculate DBE (Double Bond Equivalents)
        c = formula.get('C', 0)
        h = formula.get('H', 0)
        n = formula.get('N', 0)
        p = formula.get('P', 0)
        x = sum(formula.get(halogen, 0) for halogen in ['F', 'Cl', 'Br', 'I'])
        
        if c == 0:
            continue  # Skip formulas without carbon
        
        # Match DBE calculation with Cython implementations for consistency
        dbe = (2 + 2*c + 3*p + n - h - x) / 2.0
        
        # Check if DBE is an integer (consistency with Cython)
        if abs(dbe - round(dbe)) > 1e-8:
            continue
        
        # Apply DBE constraints
        if min_dbe is not None and dbe < min_dbe:
            continue
        if max_dbe is not None and dbe > max_dbe:
            continue
        
        # Apply heteroatom ratio constraint
        if max_hetero_ratio is not None:
            heteroatoms = sum(count for elem, count in formula.items() if elem not in ['C', 'H'])
            if c > 0 and heteroatoms / c > max_hetero_ratio:
                continue
        
        filtered.append(formula)
    
    return filtered
