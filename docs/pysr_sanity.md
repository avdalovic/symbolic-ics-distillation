# PySR Sanity Checks

Symbolic regression searches for a compact equation that maps input variables to a target. Instead of choosing a fixed model form ahead of time, PySR searches over algebraic expressions built from allowed operators.

The Pareto front is the set of candidate equations that trade off accuracy and simplicity. A more complex equation should only remain on the front if it improves the loss enough to justify the extra structure.

Complexity is PySR's size measure for an expression, usually counting variables, constants, and operators. Loss is the prediction error of an equation on the fit data. Score is PySR's measure of how much loss improves as complexity increases; it helps select useful equations from the Pareto front.

The linear sanity test checks whether PySR can recover a simple law using all three variables. This matters because later exports will include many neural-model inputs, and we need PySR to keep physically relevant variables when the relationship is simple.

The nonlinear sanity test checks whether PySR can recover an `x2*x3` interaction. This matters because industrial-control variables often interact multiplicatively through flow, level, pressure, and valve effects.

For the later LIT101 experiment, these sanity checks validate the PySR environment and our equation-selection workflow before using exported SWaT neural-model predictions.
