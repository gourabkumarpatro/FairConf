The working version of the programs were successfully tested on the combination
of the following platforms:
A) Debian GNU/Linux 10 on AMD64 architecture
B) Python 2.7.16

+=======================+
| PYTHON PACKAGES USED: |
+=======================+
argparse==1.1
cvxpy==1.0.21
gurobipy==9.1.1
matplotlib==2.2.4
numpy==1.16.2
scikit-learn==0.20.3
scipy==1.2.2

+==================+
| FOLDER STRUCTURE |
+==================+

FairConf/
├── README.txt              %% This file (tells about salient features)
├── data                    %% Contains Interest (V) & Availability (A) matrices
│   │                       %% <dataName>_V.csv: Int. matrix of size (#participants x #talks)
│   │                       %% <dataName>_A.csv: Avl. matrix of size (#participants x #slots)
│   ├── fatrecContInterest_A.csv
│   ├── fatrecContInterest_V.csv
│   ├── fatrec_A.csv
│   ├── fatrec_V.csv
│   ├── icml_A.csv
│   ├── icml_V.csv
│   ├── recsys_A.csv
│   ├── recsys_V.csv
│   ├── syn10_A.csv
│   └── syn10_V.csv
├── results                 %% Contains schedule matrices & performance metrics
│   │                       %% For each dataset, separate performance & schedules folders
│   │                       %% GENERATED SCHEDULES: <dataName>_<lam>_<lam1>_<lam2><_ILP?>_\
│   │                       %%                       CN<numProfiles>_FB1_PrioLvls<numPrios>.csv
│   │                       %%                      Schedules are 0-1 matrices of size 
│   │                       %%                      (#talks x #slots), 1 => a slot-talk map
│   │                       %% PERFORMANCE        : <allLambdaResults/singleResult>_<dataName>_... .txt
│   ├── fatrecContInterest
│   │   ├── performance
│   │   │   ├── allLambdaResults_fatrecContInterest_ILP_CN40_FB1_PrioLvls1.txt
│   │   │   ├── allLambdaResults_fatrecContInterest_ILP_CN40_FB1_PrioLvls3.txt
│   │   │   ├── singleResult_fatrecContInterest_1.0_0.5_0.5_ILP_CN40_FB1_PrioLvls1.txt
│   │   │   └── singleResult_fatrecContInterest_offlineConf.txt
│   │   └── schedules
│   │       ├── fatrecContInterest_0.0_0.0_1.0_ILP_CN40_FB1_PrioLvls1.csv
│   │       ├── . . .
│   │       ├── fatrecContInterest_1.0_1.0_0.5_ILP_CN40_FB1_PrioLvls1.csv
│   │       ├── fatrecContInterest_0.0_0.0_1.0_ILP_CN40_FB1_PrioLvls3.csv
│   │       ├── . . .
│   │       ├── fatrecContInterest_1.0_1.0_0.5_ILP_CN40_FB1_PrioLvls3.csv
│   │       ├── fatrecContInterest_IAM.csv
│   │       └── fatrecContInterest_offlineConf.csv
│   ├── icml
│   │   ├── performance
│   │   │   ├── allLambdaResults_icml_CN100_FB1_PrioLvls1_evalOnPrioLvls4.txt
│   │   │   ├── allLambdaResults_icml_CN100_FB1_PrioLvls4.txt
│   │   │   ├── singleResult_icml_1.0_0.5_0.5_CN100_FB1_PrioLvls1.txt
│   │   │   └── singleResult_icml_1.0_0.5_0.5_CN100_FB1_PrioLvls4.txt
│   │   └── schedules
│   │       ├── icml_0.0_0.0_1.0_CN100_FB1_PrioLvls1.csv
│   │       ├── . . .
│   │       ├── icml_1.0_1.0_0.5_CN100_FB1_PrioLvls1.csv
│   │       ├── icml_0.0_0.0_1.0_CN100_FB1_PrioLvls4.csv
│   │       ├── . . .
│   │       ├── icml_1.0_1.0_0.5_CN100_FB1_PrioLvls4.csv
│   │       └── icml_IAM.csv
│   ├── recsys
│   │   ├── performance
│   │   │   ├── allLambdaResults_recsys_CN1112_FB1_PrioLvls1.txt
│   │   │   ├── allLambdaResults_recsys_CN1112_FB1_PrioLvls3.txt
│   │   │   ├── singleResult_recsys_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.txt
│   │   │   ├── singleResult_recsys_P1P2P3P1P2P1_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.txt
│   │   │   ├── singleResult_recsys_P1P2P3P1P2P3_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.txt
│   │   │   ├── singleResult_recsys_P1P2P3P1P2_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.txt
│   │   │   ├── singleResult_recsys_P1P2P3P1_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.txt
│   │   │   ├── singleResult_recsys_P1P2P3_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.txt
│   │   │   └── singleResult_recsys_TwoRoundFullScale_1.0_0.05_0.05_CN1112_FB1_PrioLvls1_evalOnPrioLvls3.txt
│   │   └── schedules
│   │       ├── recsys_0.0_0.0_1.0_CN1112_FB1_PrioLvls1.csv
│   │       ├── . . .
│   │       ├── recsys_1.0_1.0_0.5_CN1112_FB1_PrioLvls1.csv
│   │       ├── recsys_0.0_0.0_1.0_CN1112_FB1_PrioLvls3.csv
│   │       ├── . . .
│   │       ├── recsys_1.0_1.0_0.5_CN1112_FB1_PrioLvls3.csv
│   │       ├── recsys_IAM.csv
│   │       ├── recsys_P1P2P3P1P2P1_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.csv
│   │       ├── recsys_P1P2P3P1P2P3_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.csv
│   │       ├── recsys_P1P2P3P1P2_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.csv
│   │       ├── recsys_P1P2P3P1_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.csv
│   │       ├── recsys_P1P2P3_1.0_0.05_0.05_CN1112_FB1_PrioLvls3.csv
│   │       ├── . . .
│   │       └── recsys_TwoRoundFullScale_1.0_0.05_0.05_CN1112_FB1_PrioLvls1.csv
│   └── syn10
│       ├── performance
│       │   ├── allLambdaResults_syn10_ILP_CN10_FB1_PrioLvls1.txt
│       │   └── singleResult_syn10_1.0_0.5_0.5_ILP_CN10_FB1_PrioLvls1.txt
│       └── schedules
│           ├── syn10_0.0_0.0_1.0_ILP_CN10_FB1_PrioLvls1.csv
│           ├── . . .
│           ├── syn10_1.0_1.0_0.5_ILP_CN10_FB1_PrioLvls1.csv
│           └── syn10_IAM.csv
└── src                     %% Contains all source codes
    ├── dataFormation       %% Contains example codes to generate A,V matrix for ICML, RecSys, FATREC
    │   ├── fatrecContInterest_Vdata.py
    │   ├── icml_data.py
    │   └── recsys_data.py
    ├── eval_multiple.py                  %% Python code to evaluate multiple conference schedules
    ├── eval_single.py                    %% Python code to evaluate a single conference schedule
    ├── eval_tools.py                     %% Python functions for schedule evaluation
    ├── optimization_ILP_LPRnd.py         %% Python code for fair-conference-scheduling
    ├── genSchedule.sh                    %% Shell script to generate a single conference schedule
    ├── genScheduleAndEval.sh             %% Shell script to generate a single conf schedule & evaluate it
    ├── genScheduleAndEval_allLambdas.sh  %% Shell script to generate set of conf schedules & evaluate all
    └── stepWiseExperiments
        └── STEPWISEoptimization_ILP_LPRnd.py

+===============+
| PRELIMINARIES |
+===============+
Considering <dataName> as the name of a dataset/conference, keep two files in ./data/ folder viz.
<dataName>_V.csv  and <dataName>_A.csv
1. <dataName>_V.csv is the Interest matrix of size (#participants x #talks). Each cell (i,j)
   contains a positive fractional value. It indicates how interested is participant i to attend
   the j-th  talk. 
2. <dataName>_A.csv is the Availability matrix of size (#participants x #slots). Each cell (i,j)
   contains a positive fractional value. It indicates how much available is participant i 
   in the j-th slot.

+========================+
| EXECUTION INSTRUCTIONS |
+========================+
<dataName>    => name of dataset/conference                   (any alphanumeric string)
<lamb>        => determines weight of efficiency              (any real value between 0 and 1)
<lam1>        => determines weight of participant fairness    (any real value between 0 and 1)
<lam2>        => determines weight of speaker fairness        (any real value between 0 and 1)
<confType>    => size of conference                           (1 = small-scale, 2 = medium-scale, 
                                                               3 = large-scale)
<clusterFrac> => fraction to which participants are clustered (any positive real value less than 1,
                                                               ~0 = one cluster, 1.0 = no clustering)
<n_clusters>  => no. of unique participant profiles           (any positive integer,
                                                               min = 1, max = #participants)
<numPrios>    => number of priority-levels                    (any positive integer, < #talks,
                                                               if <numPrios> = 3, all talks grouped
                                                               into 3 prio levels of equal size)
                                                               
1. For small-scale (<confType> = 1) or 
       medium-scale (<confType> = 2) conferences:
--------------------------------------------------
a) To generate a single conference schedule:
   >> ./src/genSchedule.sh <dataName> <lam> <lam1> <lam2> --confType <confType> 
                                                         [--num_prio_lvl <numPrios> 
                                                          OR --num_prio_lvl_EVALONLY <numPrios>]
b) To generate a single conference schedule and evaluate it:
   >> ./src/genScheduleAndEval.sh <dataName> <lam> <lam1> <lam2> --confType <confType> 
                                                         [--num_prio_lvl <numPrios> 
                                                          OR --num_prio_lvl_EVALONLY <numPrios>]
c) To generate a set of conference schedules with varying lambdas and evaluate them all:
   >> ./src/genScheduleAndEval_allLambdas.sh <dataName> --confType <confType> 
                                                         [--num_prio_lvl <numPrios> 
                                                          OR --num_prio_lvl_EVALONLY <numPrios>]
   
2. For large-scale conferences (<confType> = 3):
------------------------------------------------
a) To generate a single conference schedule:
   >> ./src/genSchedule.sh <dataName> <lam> <lam1> <lam2> --confType <confType> 
                                                         [--num_prio_lvl <numPrios> 
                                                          OR --num_prio_lvl_EVALONLY <numPrios>]
                                                         [--n_clusters <n_clusters>
                                                          OR --clusterFrac <clusterFrac>]
b) To generate a single conference schedule and evaluate it:
   >> ./src/genScheduleAndEval.sh <dataName> <lam> <lam1> <lam2> --confType <confType> 
                                                         [--num_prio_lvl <numPrios> 
                                                          OR --num_prio_lvl_EVALONLY <numPrios>]
                                                         [--n_clusters <n_clusters>
                                                          OR --clusterFrac <clusterFrac>] 
c) To generate a set of conference schedules with varying lambdas and evaluate them all:
   >> ./src/genScheduleAndEval_allLambdas.sh <dataName> --confType <confType> 
                                                         [--num_prio_lvl <numPrios> 
                                                          OR --num_prio_lvl_EVALONLY <numPrios>]
                                                         [--n_clusters <n_clusters>
                                                          OR --clusterFrac <clusterFrac>]

3. For evaluating a schedule:
----------------------------
First, place schedule at path results/<dataName>/schedules/<nameOfScheduleCSVfile_withoutExtn>.csv
The schedule in the file should be a 0-1 matrix (X) of size (#talks x #slots).
It is a slot-talk map, where X[t,s] = 1 indicates talk t is scheduled in slot s.
Then, run the following commands (Use same argument set in 1 and 3):

1. Create baseline schedules (Generates three baseline schedules at results/<dataName>/schedules/)
   >> ./src/genScheduleAndEval.sh <dataName> 0.0 0.0 1.0 --confType <confType> 
                                                         [--num_prio_lvl <numPrios> 
                                                          OR --num_prio_lvl_EVALONLY <numPrios>]
                                                         [--n_clusters <n_clusters>
                                                          OR --clusterFrac <clusterFrac>]
2. Change directory
   >> cd ./src
3. Evaluate custom schedule
   >> python eval_single.py --dataset <dataName> --confType <confType> 
                                                 --scheduleName_overrided <nameOfScheduleCSVfile_withoutExtn>
                                                 [--num_prio_lvl <numPrios> 
                                                  OR --num_prio_lvl_EVALONLY <numPrios>]
                                                 [--n_clusters <n_clusters>
                                                  OR --clusterFrac <clusterFrac>]

+=========+
| EXAMPLE |
+=========+
(1) syn10 is a small-scale synthetic conference
    The schedule "syn10_0.0_0.0_1.0_ILP_CN10_FB1_PrioLvls1.csv" to 
    "syn10_1.0_1.0_0.5_ILP_CN10_FB1_PrioLvls1.csv", along with performance file
    "allLambdaResults_syn10_ILP_CN10_FB1_PrioLvls1.txt" can be generated by:

    >> ./src/genScheduleAndEval_allLambdas.sh syn10 --confType 1
    
(2) Individual schedule "syn10_0.0_0.0_1.0_ILP_CN10_FB1_PrioLvls1.csv" (where weights 
    of efficiency, participant fairness and speaker fairness are 1.0, 0.5, 0.5 resp.)
    along with performance file 
    "singleResult_syn10_1.0_0.5_0.5_ILP_CN10_FB1_PrioLvls1.txt" can be generated by:
    
    >> ./src/genScheduleAndEval.sh syn10 1.0 0.5 0.5 --confType 1
    
(3) fatrecContInterest is a small-scale real conference. So, schedules can be generated like syn10.
    Further, existing schedules e.g. fatrecContInterest_offlineConf.csv can be evaluated by:
    
    >> ./src/genScheduleAndEval.sh fatrecContInterest 0.0 0.0 1.0 --confType 1
    >> cd ./src
    >> python eval_single.py --dataset fatrecContInterest --confType 1 
                             --scheduleName_overrided fatrecContInterest_offlineConf
                             
(4) recsys is a medium-scale real conference. Schedules 
    "recsys_0.0_0.0_1.0_CN1112_FB1_PrioLvls1.csv" to 
    "recsys_1.0_1.0_0.5_CN1112_FB1_PrioLvls1.csv", along with performance file
    "allLambdaResults_recsys_CN1112_FB1_PrioLvls1.txt" can be generated by:

    >> ./src/genScheduleAndEval_allLambdas.sh recsys --confType 2
    
(5) Scheduling can be done priority-wise.
    If we intend to group the talks into three priority-levels, where Prio-1 (high) talks
    scheduled thrice, Prio-2 talks scheduled twice and Prio-3 (low) talks scheduled once,
    until there remains sufficient slots to allot:
    
    >> ./src/genScheduleAndEval_allLambdas.sh recsys --confType 2 --num_prio_lvl 3
    
(6) If we do not intend to prioritize talks during scheduling, but during evaluation
    we want speaker-side metrics to be computed for each priority-level:
    
    >> ./src/genScheduleAndEval_allLambdas.sh recsys --confType 2 --num_prio_lvl_EVALONLY 3
    
(7) icml is a large-scale conference. For 100 clustered participant-profiles, schedules
    "icml_0.0_0.0_1.0_CN100_FB1_PrioLvls4.csv" to 
    "icml_1.0_1.0_0.5_CN100_FB1_PrioLvls4.csv", along with performance file
    "singleResult_icml_1.0_0.5_0.5_CN100_FB1_PrioLvls4.txt" can be generated by:
    
    >> ./src/genScheduleAndEval_allLambdas.sh icml --confType 3 --num_prio_lvl 4 --n_clusters 100
    
(8) If we do not intend to prioritize talks, for large conferences like icml we can use:
 
    >> ./src/genScheduleAndEval_allLambdas.sh icml --confType 3 --n_clusters 100
       (OR)
    >> ./src/genScheduleAndEval_allLambdas.sh icml --confType 3 --n_clusters 100 --num_prio_lvl_EVALONLY 4
    
    The second one is to be used when we want priority-level evaluation for 
    speaker-side metrics, using 4 priority-levels.

------- END -------