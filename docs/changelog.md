# Changelog and Bugs

December 1st, 2023. [v0.0.2]. 
- Read angular velocity at control frequency. Previously the angular velocity is obtained at simulation frequency. We found this will result in oscillation behavior since the policy will learn to exploit this phenomenon.
- Remove the hand-crafted lower and upper of privileged information.
- Remove online mass randomization since it does not have effect after simulation is created.
- Change angular velocity max clip limit from 0.5 to 0.4, to compensate the higher rotation speed.