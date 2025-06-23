from z3 import Solver, Bool, Sum, If, Int, Or, And, Not, sat, Optimize, is_true
import math
from typing import List, Dict, Literal, Optional
from time import perf_counter


def z3_solve_mass_decomposition_optimized(
    target_mass: float,
    error_ppm: float,
    element_details: Dict[str, Dict[str, float]],
    rules: list = None,
    max_solutions: int = 100,
    solver_options: dict = None,
    enable_profiling: bool = False,
    optimization_strategy: Literal["push_pop", "fresh_solver", "minimize_vars", "all_smt"] = "all_smt",
):
    """
    Optimized Z3-based mass decomposition with multiple strategies to reduce solving time.
    """
    if rules is None:
        rules = []
    if solver_options is None:
        solver_options = {}

    print(f"Target mass: {target_mass:.6f}, Error: {error_ppm} ppm")
    print(f"Optimization strategy: {optimization_strategy}")

    match optimization_strategy:
        case "push_pop":
            return _solve_with_push_pop_fixed(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling)
        case "fresh_solver":
            return _solve_with_fresh_solver(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling)
        case "minimize_vars":
            return _solve_with_minimized_vars(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling)
        case "all_smt":
            return _solve_with_all_smt(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling)

def _solve_with_push_pop_fixed(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling):
    """
    Strategy 1: Use push/pop to manage exclusion constraints efficiently
    The original approach was actually correct - Z3 maintains learned clauses between iterations
    """
    mass_tolerance_absolute = target_mass * error_ppm * 1e-6
    min_mass = target_mass - mass_tolerance_absolute
    max_mass = target_mass + mass_tolerance_absolute
    
    found_formulas = []
    element_symbols = list(element_details.keys())
    
    # Create base solver with all permanent constraints
    base_solver = Solver()
    
    # Variable creation (same as original)
    choices = {}
    count_ranges = {}
    
    for element_symbol in element_symbols:
        element_data = element_details[element_symbol]
        min_count = element_data.get("min", 0)
        max_allowed_count = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
        max_count = element_data.get("max", max_allowed_count)
        max_count = min(max_count, max_allowed_count)
        count_ranges[element_symbol] = range(min_count, max_count + 1)
        choices[element_symbol] = {}
        
        for count in count_ranges[element_symbol]:
            choices[element_symbol][count] = Bool(f"Choice_{element_symbol}_{count}")

    # Add permanent constraints to base solver
    # One-hot constraints
    for element_symbol in element_symbols:
        base_solver.add(Sum([If(choices[element_symbol][count], 1, 0) 
                           for count in count_ranges[element_symbol]]) == 1)

    # Mass constraints
    scale_factor = 1000000
    mass_contributions = []
    for element_symbol in element_symbols:
        for count in count_ranges[element_symbol]:
            scaled_mass = int(element_details[element_symbol]["mass"] * scale_factor)
            mass_contributions.append(
                If(choices[element_symbol][count], count * scaled_mass, 0)
            )
    
    total_mass_scaled = Sum(mass_contributions)
    min_mass_scaled = int(min_mass * scale_factor)
    max_mass_scaled = int(max_mass * scale_factor)
    
    base_solver.add(total_mass_scaled >= min_mass_scaled)
    base_solver.add(total_mass_scaled <= max_mass_scaled)

    # Add rules
    if rules:
        for rule_func in rules:
            try:
                import inspect
                sig = inspect.signature(rule_func)
                if len(sig.parameters) == 2:
                    constraint = rule_func(choices, count_ranges)
                else:
                    element_counts = {}
                    for element_symbol in element_symbols:
                        element_counts[element_symbol] = Sum([
                            If(choices[element_symbol][count], count, 0)
                            for count in count_ranges[element_symbol]
                        ])
                    constraint = rule_func(element_counts)
                
                if constraint is not None:
                    base_solver.add(constraint)
            except Exception as e:
                print(f"Warning: Error applying rule: {e}")

    # Solving loop - add exclusions permanently to benefit from learned clauses
    solution_count = 0
    
    while solution_count < max_solutions:
        solve_start = perf_counter()
        result = base_solver.check()
        solve_time = perf_counter() - solve_start
        
        if enable_profiling:
            print(f"PROFILE: Solve iteration {solution_count + 1} took {solve_time:.4f} seconds")
        
        if result == sat:
            model = base_solver.model()
            current_formula = {}
            actual_mass = 0.0
            
            # Extract solution
            current_solution_vars = []
            for element_symbol in element_symbols:
                element_count = 0
                for count in count_ranges[element_symbol]:
                    choice_value = model[choices[element_symbol][count]]
                    if is_true(choice_value):
                        element_count = count
                        current_solution_vars.append(choices[element_symbol][count])
                
                if element_count > 0:
                    current_formula[element_symbol] = element_count
                actual_mass += element_count * element_details[element_symbol]["mass"]
            
            delta_ppm = ((actual_mass - target_mass) / target_mass) * 1e6
            print(f"solution {solution_count + 1}: {current_formula} (mass: {actual_mass:.6f}, delta: {delta_ppm:.2f} ppm)")
            found_formulas.append(current_formula)
            solution_count += 1
            
            # Add exclusion constraint permanently - Z3 keeps learned clauses
            if current_solution_vars:
                exclusion_constraint = Not(And(current_solution_vars))
                base_solver.add(exclusion_constraint)
        else:
            break

    return found_formulas

def _solve_with_fresh_solver(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling):
    """
    Strategy 2: Create fresh solver for each iteration with accumulated exclusions
    """
    mass_tolerance_absolute = target_mass * error_ppm * 1e-6
    min_mass = target_mass - mass_tolerance_absolute
    max_mass = target_mass + mass_tolerance_absolute
    
    found_formulas = []
    element_symbols = list(element_details.keys())
    excluded_solutions = []  # Store solutions to exclude
    
    for iteration in range(max_solutions):
        # Create fresh solver
        solver = Solver()
        
        # Variable creation
        choices = {}
        count_ranges = {}
        
        for element_symbol in element_symbols:
            element_data = element_details[element_symbol]
            min_count = element_data.get("min", 0)
            max_allowed_count = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
            max_count = element_data.get("max", max_allowed_count)
            max_count = min(max_count, max_allowed_count)
            count_ranges[element_symbol] = range(min_count, max_count + 1)
            choices[element_symbol] = {}
            
            for count in count_ranges[element_symbol]:
                choices[element_symbol][count] = Bool(f"Choice_{element_symbol}_{count}")

        # Add all constraints (same as before)
        for element_symbol in element_symbols:
            solver.add(Sum([If(choices[element_symbol][count], 1, 0) 
                           for count in count_ranges[element_symbol]]) == 1)

        # Mass constraints
        scale_factor = 1000000
        mass_contributions = []
        for element_symbol in element_symbols:
            for count in count_ranges[element_symbol]:
                scaled_mass = int(element_details[element_symbol]["mass"] * scale_factor)
                mass_contributions.append(
                    If(choices[element_symbol][count], count * scaled_mass, 0)
                )
        
        total_mass_scaled = Sum(mass_contributions)
        min_mass_scaled = int(min_mass * scale_factor)
        max_mass_scaled = int(max_mass * scale_factor)
        
        solver.add(total_mass_scaled >= min_mass_scaled)
        solver.add(total_mass_scaled <= max_mass_scaled)

        # Add rules
        if rules:
            for rule_func in rules:
                try:
                    import inspect
                    sig = inspect.signature(rule_func)
                    if len(sig.parameters) == 2:
                        constraint = rule_func(choices, count_ranges)
                    else:
                        element_counts = {}
                        for element_symbol in element_symbols:
                            element_counts[element_symbol] = Sum([
                                If(choices[element_symbol][count], count, 0)
                                for count in count_ranges[element_symbol]
                            ])
                        constraint = rule_func(element_counts)
                    
                    if constraint is not None:
                        solver.add(constraint)
                except Exception as e:
                    print(f"Warning: Error applying rule: {e}")

        # FIXED: Add exclusion constraints for all previously found solutions
        for excluded_formula in excluded_solutions:
            exclusion_vars = []
            for element_symbol in element_symbols:
                count = excluded_formula.get(element_symbol, 0)
                if count in count_ranges[element_symbol]:
                    exclusion_vars.append(choices[element_symbol][count])
            
            # FIXED: Use And() + Not() to exclude this exact combination
            if exclusion_vars:
                solver.add(Not(And(exclusion_vars)))

        # Solve
        solve_start = perf_counter()
        result = solver.check()
        solve_time = perf_counter() - solve_start
        
        if enable_profiling:
            print(f"PROFILE: Solve iteration {iteration + 1} took {solve_time:.4f} seconds")
        
        if result == sat:
            model = solver.model()
            current_formula = {}
            actual_mass = 0.0
            
            for element_symbol in element_symbols:
                element_count = 0
                for count in count_ranges[element_symbol]:
                    choice_value = model[choices[element_symbol][count]]
                    if is_true(choice_value):
                        element_count = count
                
                if element_count > 0:
                    current_formula[element_symbol] = element_count
                actual_mass += element_count * element_details[element_symbol]["mass"]
            
            delta_ppm = ((actual_mass - target_mass) / target_mass) * 1e6
            print(f"solution {iteration + 1}: {current_formula} (mass: {actual_mass:.6f}, delta: {delta_ppm:.2f} ppm)")
            found_formulas.append(current_formula)
            excluded_solutions.append(current_formula)
        else:
            break

    return found_formulas

def _solve_with_minimized_vars(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling):
    """
    Strategy 3: Use integer variables instead of one-hot binary variables
    """
    mass_tolerance_absolute = target_mass * error_ppm * 1e-6
    min_mass = target_mass - mass_tolerance_absolute
    max_mass = target_mass + mass_tolerance_absolute
    
    found_formulas = []
    element_symbols = list(element_details.keys())
    
    # Create solver with integer variables (much fewer variables)
    solver = Solver()
    element_counts = {}
    
    for element_symbol in element_symbols:
        element_data = element_details[element_symbol]
        min_count = element_data.get("min", 0)
        max_allowed_count = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
        max_count = element_data.get("max", max_allowed_count)
        max_count = min(max_count, max_allowed_count)
        
        element_counts[element_symbol] = Int(f"count_{element_symbol}")
        solver.add(element_counts[element_symbol] >= min_count)
        solver.add(element_counts[element_symbol] <= max_count)

    # Mass constraints
    scale_factor = 1000000
    total_mass_scaled = Sum([element_counts[element_symbol] * int(element_details[element_symbol]["mass"] * scale_factor)
                            for element_symbol in element_symbols])
    min_mass_scaled = int(min_mass * scale_factor)
    max_mass_scaled = int(max_mass * scale_factor)
    
    solver.add(total_mass_scaled >= min_mass_scaled)
    solver.add(total_mass_scaled <= max_mass_scaled)

    # Add rules (adapted for integer variables)
    if rules:
        for rule_func in rules:
            try:
                constraint = rule_func(element_counts)
                if constraint is not None:
                    solver.add(constraint)
            except Exception as e:
                print(f"Warning: Error applying rule: {e}")

    # Solving loop
    solution_count = 0
    
    while solution_count < max_solutions:
        solve_start = perf_counter()
        result = solver.check()
        solve_time = perf_counter() - solve_start
        
        if enable_profiling:
            print(f"PROFILE: Solve iteration {solution_count + 1} took {solve_time:.4f} seconds")
        
        if result == sat:
            model = solver.model()
            current_formula = {}
            actual_mass = 0.0
            
            for element_symbol in element_symbols:
                count = model[element_counts[element_symbol]]
                element_count = count.as_long() if count is not None else 0
                
                if element_count > 0:
                    current_formula[element_symbol] = element_count
                actual_mass += element_count * element_details[element_symbol]["mass"]
            
            delta_ppm = ((actual_mass - target_mass) / target_mass) * 1e6
            print(f"solution {solution_count + 1}: {current_formula} (mass: {actual_mass:.6f}, delta: {delta_ppm:.2f} ppm)")
            found_formulas.append(current_formula)
            solution_count += 1
            
            # FIXED: Add exclusion constraint for this exact solution
            exclusion_constraint = Not(And([element_counts[element_symbol] == current_formula.get(element_symbol, 0)
                                           for element_symbol in element_symbols]))
            solver.add(exclusion_constraint)
        else:
            break

    return found_formulas

def _solve_with_all_smt(target_mass, error_ppm, element_details, rules, max_solutions, enable_profiling):
    """
    Strategy 4: Use optimal all_smt algorithm with divide-and-conquer approach
    This is the most efficient method for finding multiple solutions.
    """
    mass_tolerance_absolute = target_mass * error_ppm * 1e-6
    min_mass = target_mass - mass_tolerance_absolute
    max_mass = target_mass + mass_tolerance_absolute
    
    found_formulas = []
    element_symbols = list(element_details.keys())
    
    # Create solver with integer variables (simpler approach)
    solver = Solver()
    element_counts = {}
    
    for element_symbol in element_symbols:
        element_data = element_details[element_symbol]
        min_count = element_data.get("min", 0)
        max_allowed_count = math.floor(max_mass / element_data["mass"]) if element_data["mass"] > 0 else 0
        max_count = element_data.get("max", max_allowed_count)
        max_count = min(max_count, max_allowed_count)
        
        element_counts[element_symbol] = Int(f"count_{element_symbol}")
        solver.add(element_counts[element_symbol] >= min_count)
        solver.add(element_counts[element_symbol] <= max_count)

    # Mass constraints
    scale_factor = 1000000
    total_mass_scaled = Sum([element_counts[element_symbol] * int(element_details[element_symbol]["mass"] * scale_factor)
                            for element_symbol in element_symbols])
    min_mass_scaled = int(min_mass * scale_factor)
    max_mass_scaled = int(max_mass * scale_factor)
    
    solver.add(total_mass_scaled >= min_mass_scaled)
    solver.add(total_mass_scaled <= max_mass_scaled)

    # Add rules (adapted for integer variables)
    if rules:
        for rule_func in rules:
            try:
                constraint = rule_func(element_counts)
                if constraint is not None:
                    solver.add(constraint)
            except Exception as e:
                print(f"Warning: Error applying rule: {e}")

    # Get all variables for the all_smt algorithm
    variables = [element_counts[symbol] for symbol in element_symbols]
    
    def block_term(s, m, t):
        """Block a specific evaluation of term t in model m"""
        s.add(t != m.eval(t, model_completion=True))
        
    def fix_term(s, m, t):
        """Fix a specific evaluation of term t in model m"""
        s.add(t == m.eval(t, model_completion=True))
        
    def all_smt_rec(terms):
        """Recursive function to enumerate all models using divide-and-conquer"""
        solve_start = perf_counter()
        if solver.check() == sat:
            solve_time = perf_counter() - solve_start
            if enable_profiling:
                print(f"PROFILE: all_smt iteration took {solve_time:.4f} seconds")
                
            m = solver.model()
            yield m
            
            # Divide and conquer: for each term, block it and fix previous terms
            for i in range(len(terms)):
                solver.push()
                block_term(solver, m, terms[i])
                for j in range(i):
                    fix_term(solver, m, terms[j])
                yield from all_smt_rec(terms[i:])
                solver.pop()
    
    # Use the all_smt algorithm to enumerate solutions
    solution_count = 0
    for model in all_smt_rec(variables):
        if solution_count >= max_solutions:
            break
            
        current_formula = {}
        actual_mass = 0.0
        
        for element_symbol in element_symbols:
            count = model[element_counts[element_symbol]]
            element_count = count.as_long() if count is not None else 0
            
            if element_count > 0:
                current_formula[element_symbol] = element_count
            actual_mass += element_count * element_details[element_symbol]["mass"]
        
        delta_ppm = ((actual_mass - target_mass) / target_mass) * 1e6
        print(f"solution {solution_count + 1}: {current_formula} (mass: {actual_mass:.6f}, delta: {delta_ppm:.2f} ppm)")
        found_formulas.append(current_formula)
        solution_count += 1

    return found_formulas

# Updated rule functions for integer variables
def rule_DBE_int(element_counts):
    """DBE rule for integer variables"""
    if "H" not in element_counts or "C" not in element_counts:
        raise ValueError("Element counts must include 'H' and 'C' for DBE rule")
    return element_counts["H"] <= 2 * element_counts["C"] + 3

def rule_oc_ratio_int(element_counts):
    """O/C ratio rule for integer variables"""
    if "O" not in element_counts or "C" not in element_counts:
        raise ValueError("Element counts must include 'O' and 'C' for O/C ratio rule")
    return element_counts["O"] <= element_counts["C"]

if __name__ == "__main__":
    elements_data_generic = {
        "C": {"mass": 12.000000, "min": 14, "max": 20},
        "H": {"mass": 1.007825, "min": 0, "max": 50},
        "O": {"mass": 15.994914, "min": 0, "max": 20},
        "N": {"mass": 14.003074, "min": 0, "max": 10},
        "S": {"mass": 31.972071, "min": 0, "max": 3},
        "P": {"mass": 30.973762, "min": 0, "max": 5},
        "Na": {"mass": 22.989769, "min": 0, "max": 1},
        "F": {"mass": 18.998403, "min": 0, "max": 20},
        "Cl": {"mass": 34.968853, "min": 0, "max": 0},
        "Br": {"mass": 78.918337, "min": 0, "max": 0},
        "I": {"mass": 126.904473, "min": 0, "max": 2},
    }

    measured_mass_user = 285.136493
    error_user = 2

    # Test different strategies
    strategies = ["all_smt", "minimize_vars", "fresh_solver", "push_pop"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy}")
        print(f"{'='*60}")
        
        start_time = perf_counter()
        if strategy in ["minimize_vars", "all_smt"]:
            rules = [rule_DBE_int, rule_oc_ratio_int]
        else:
            rules = [rule_DBE_int, rule_oc_ratio_int]
            
        solutions = z3_solve_mass_decomposition_optimized(
            measured_mass_user,
            error_user,
            elements_data_generic,
            rules=rules,
            max_solutions=20,
            enable_profiling=True,
            optimization_strategy=strategy,
        )
        
        total_time = perf_counter() - start_time
        print(f"\nStrategy '{strategy}': Found {len(solutions)} solutions in {total_time:.4f}s")
