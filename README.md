# CAPBO
Data available for the AIChE paper

Branch: indicates data uploaded on different dates

Naming rules for the charts:

'Sensitivity': serial experiments; 'Parallel': parallel experiments;
'Hartman3', 'Alpine', 'Ackley', 'Hartman6': benchmark tests for different functions;
'High', 'low': the optimum is in the high-cost/low-cost region;
'2dim', '4dim', '6dim': case studies of different dimensional problems;
'gls': case study of hydrogenation of m-dinitrobenzene.

Naming rules for the sheets:

Template: 'A_B'
A: the acquisition function used for this sheet;
B: the record for time or the current optimum (represented by 't' or 'result');
Each column corresponds to a single experimental results.



Code availability

Related source code will come soon, and a simple demo is provided temporarily for easier understanding.

Dependence: Bayesian-optimization package v=1.2.0 https://github.com/bayesian-optimization/BayesianOptimization/releases/tag/1.2.0

Please overwrite the files in the original package with the provided ones (bayesian_optimization.py, util.py, and target_space.py). Then run demo.py.

For the results of SNaKe, please refer to https://github.com/cog-imperial/SnAKe.
