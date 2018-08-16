# Chicxulub_StressStrain
This repository contains all data presented within "Stress-Strain Evolution during Peak-Ring Formation: A Case Study of the Chicxulub Impact Structure" by Auriol S. P. Rae, Gareth S. Collins, Michael Poelchau, Ulrich Riller, Thomas M. Davison, Richard A. F. Grieve, Gordon R. Osinski, Joanna V. Morgan, IODP-ICDP Expedition 364 Scientists

There are 4 folders within this directory: iSALE_inputs, StressStrain_Calculator, Model_Comparison, and Tracer_Comparison.

iSALE_inputs:
This folder provides all of the iSALE input files required to produce the models discussed in the contribution.
  - 'Rae_18_prime' is the model used for all of the analysis presented within the contribution. The contribution used three runs of the model, the only difference between the runs being that the time between saved steps was decreased to produce greater temporal resolution when it was required (such as for producing results for the deformation during shock). The two .inp files within the directory 'Chicxulub_StressStrain/iSALE_inputs/Rae_18_prime' possess the lowest temporal resolution, with 2 s between timesteps, finishing at 600 s. The two additional folders within 'Chicxulub_StressStrain/iSALE_inputs/Rae_18_prime'; '1s_res' and '10us_res', respectively contain the input files for a run with a temporal resolution of 1 s between timesteps, finishing at 300 s; and a run with a temporal resolution of 10 micro-s between timesteps, finishing at 6 s.
  - 'Rae_18_alternative' provides an alternative model to the one presented in the contribution, where the impactor diameter is 14 km, rather than 12 km. All other parameters remain the same.
  
StressStrain_Calculator:
This folder contains a commented python script that can be used on any jdata.dat file produced by iSALE2D. It requires that the user has saved the Tracer Fields: ['Trd', 'Trp', 'TrX', 'TrY', 'TrR', 'TrH', 'Trt', 'TrA']  in their iSALE model. The script has indicators to show where a user may need to make edits with regard to their own file directories. The developer of this tool, Auriol Rae, would appreciate acknowledgment of its use and the reporting of any issues in the code.

Model_Comparison:
Here, three iSALE simulations of the Chicxulub impact event are compared, both between themselves (F1), and with a geophysically constrained cross-section of the Chicxulub structure based on Vermeesch and Morgan (2008) (F2). In summary, the updated simulation presented in this work (F1a; 'Rae_18_prime') is not substantially different to previously published models (e.g. Figure F1c; Morgan et al. 2016). 'Rae_18_alternative' is shown in panels F1b and F2b.  Nevertheless, there are several distinct improvements:
  1. The impact velocity has been increased to a more realistic value of 15 km/s.
  2. The depth of the modelled crater is considerably closer to the true depth of the Chicxulub Crater, although the model remains around 400 m too deep. At this point, it is worth noting that all published numerical impact simulations predict craters that are too deep (see references within the main contribution).
  3. The impact melt volume remains consistent with the geophysically constrained volume.
  4. The topography of the rim of the crater is less extreme.
  5. The new model has increased spatial resolution.

Tracer_Comparison:
Here, data from the selected Lagrangian pseudo-cell in the contribution are compared to other Lagrangian pseudo-cells within the Chicxulub peak-ring. F1 shows the relative locations of the compared tracers, while F2 and F3 respectively show the strain and stress histories of the Lagrangian pseudo-cells seen in F1. In summary, the Lagrangian pseudo-cell chosen in the contribution represents an “average” parcel of peak-ring material. The patterns of stress-strain evolution in all pseudo-cells are qualitatively similar, where deformation in a pseudo-cell may be displaced temporally to another pseudo-cell, as a consequence of the spatial difference between them. Additionally, there are some quantitative variations between pseudo-cells that vary systematically, associated with location in the peak-ring e.g. cumulative rotation, which varies by ± 30◦.
