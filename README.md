# Chicxulub_StressStrain
This repository contains the Additional Supplementary Material associated with "Stress-Strain Evolution during Peak-Ring Formation: A Case Study of the Chicxulub Impact Structure" by Auriol S. P. Rae, Gareth S. Collins, Michael Poelchau, Ulrich Riller, Thomas M. Davison, Richard A. F. Grieve, Gordon R. Osinski, Joanna V. Morgan, IODP-ICDP Expedition 364 Scientists

There are 2 folders within this directory: iSALE_inputs, and StressStrain_Calculator.

iSALE_inputs:
This folder provides all of the iSALE input files required to produce the models discussed in the contribution.
  - 'Rae_18_prime' is the model used for all of the analysis presented within the contribution. The contribution used three runs of the model, the only difference between the runs being that the time between saved steps was decreased to produce greater temporal resolution when it was required (such as for producing results for the deformation during shock). The two .inp files within the directory 'Chicxulub_StressStrain/iSALE_inputs/Rae_18_prime' possess the lowest temporal resolution, with 2 s between timesteps, finishing at 600 s. The two additional folders within 'Chicxulub_StressStrain/iSALE_inputs/Rae_18_prime'; '1s_res' and '10us_res', respectively contain the input files for a run with a temporal resolution of 1 s between timesteps, finishing at 300 s; and a run with a temporal resolution of 10 micro-s between timesteps, finishing at 6 s.
  - 'Rae_18_alternative' provides an alternative model to the one presented in the Supplementary Material of the contribution, where the impactor diameter is 14 km, rather than 12 km. All other parameters remain the same.
  
StressStrain_Calculator:
This folder contains a commented python script that can be used on any jdata.dat file produced by iSALE2D. It requires that the user has saved the Tracer Fields: ['Trd', 'Trp', 'TrX', 'TrY', 'TrR', 'TrH', 'Trt', 'TrA']  in their iSALE model. The script has indicators to show where a user may need to make edits with regard to their own file directories. The developer of this tool, Auriol Rae, would appreciate acknowledgment of its use and the reporting of any issues in the code.
