from eval_tools import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluating Virtual Conference Scheduling')
    parser.add_argument('--dataset', dest='dataset', type=str,
                      help='name of dataset/conference')
    parser.add_argument('--confType', dest='confType', type=int, default=3, 
                      choices=[1, 2, 3], help='size of conference, 1=small, 2=mid, 3=large')
    parser.add_argument('--clusterFrac', dest='clusterFrac', type=float, default=None,
                      help='fraction to which participants are clustered, ~0 = one cluster, 1.0 = no clustering')
    parser.add_argument('--n_clusters', dest='n_clusters', type=int, default=None,
                      help='no. of unique participant profiles')
    parser.add_argument('--num_prio_lvl', dest='num_prio_lvl', type=int, default=None,
                      help='no. of priority levels')
    parser.add_argument('--fairBoundIndx', dest='fairBoundIndx', type=int, default=1,
                      help='to do away with outliers in obj_fn. ' +\
                           'fairBoundIndx=1 means max & min, =2 means 2nd max & 2nd min')
    parser.add_argument('--num_prio_lvl_EVALONLY', dest='num_prio_lvl_EVALONLY', type=int, default=None,
                      help='flag to calc metrics similar to prio, even if schedule is not of prioritized talks')

    args = parser.parse_args()
    if (args.clusterFrac is not None) and (args.n_clusters is not None):
        print ("Both clusterFrac and n_clusters cannot be supplied. Please supply only one of them.")
        sys.exit(0)
    if (args.num_prio_lvl_EVALONLY is not None):
        args.num_prio_lvl = args.num_prio_lvl_EVALONLY
    return args

def print_optimization_results(V, A, dataset, resultDir, instanceSuffix):
    global lamda_vals, args, V_partitioned_list, V_partitioned_indxList
    mets=["NCG_mean", "NCG_max-NCG_min", "NCG_Gini", "NEC_mean", "NEC_max-NEC_min", "NEC_Gini", "TEP"]
    partitionwise_mets = ["NEC_mean", "NEC_max-NEC_min", "NEC_Gini"]
    modifiedPrint("lamda_values " + str(lamda_vals))
    
    # --- Lamda-1 Varied Results, when Lamda-2 = 0.5 ---
    metVal_listOfDict = []
    for lamda_1 in lamda_vals:
        X=np.loadtxt(resultDir + "/" + dataset + \
                     "_1.0_" + str(lamda_1) + "_0.5" + instanceSuffix + ".csv", delimiter=",")
        metVal_listOfDict.append(metrics(V, A, X, args.num_prio_lvl, V_partitioned_list, V_partitioned_indxList))

    for met in mets:
        if (args.num_prio_lvl is not None) and (met in partitionwise_mets):
            for partn_key in collections.OrderedDict(sorted(metVal_listOfDict[0][met].items())): #getting keys in sorted order
                modifiedPrint("=================================== " + dataset + " lamda_1 (lamda = 1, lamda_2 = 0.5) === " +
                               str(met) + "_" + partn_key + " ===")
                for lambda_1_indx, lamda_1 in enumerate(lamda_vals):
                    modifiedPrint("(" + str(lamda_1) + ",\t" + str(metVal_listOfDict[lambda_1_indx][met][partn_key]) + str(")"))
        else:
            modifiedPrint("=================================== " + dataset +\
                          " lamda_1 (lamda = 1, lamda_2 = 0.5) === " + str(met) + " ===")
            for lambda_1_indx, lamda_1 in enumerate(lamda_vals):
                modifiedPrint("(" + str(lamda_1) + ",\t" + str(metVal_listOfDict[lambda_1_indx][met]) + str(")"))
            
    # --- Lamda-2 Varied Results, when Lamda-1 = 0.5 ---
    metVal_listOfDict = []
    for lamda_2 in lamda_vals:
        X=np.loadtxt(resultDir + "/" + dataset + \
                         "_1.0_0.5_" + str(lamda_2) + instanceSuffix + ".csv", delimiter=",")
        metVal_listOfDict.append(metrics(V, A, X, args.num_prio_lvl, V_partitioned_list, V_partitioned_indxList))

    for met in mets:
        if (args.num_prio_lvl is not None) and (met in partitionwise_mets):
            for partn_key in collections.OrderedDict(sorted(metVal_listOfDict[0][met].items())): #getting keys in sorted order
                modifiedPrint("=================================== " + dataset + " lamda_2 (lamda = 1, lamda_1 = 0.5) === " +
                               str(met) + "_" + partn_key + " ===")
                for lambda_2_indx, lamda_2 in enumerate(lamda_vals):
                    modifiedPrint("(" + str(lamda_2) + ",\t" + str(metVal_listOfDict[lambda_2_indx][met][partn_key]) + str(")"))
        else:
            modifiedPrint("=================================== " + dataset +\
                          " lamda_2 (lamda = 1, lamda_1 = 0.5) === " + str(met) + " ===")
            for lambda_2_indx, lamda_2 in enumerate(lamda_vals):
                modifiedPrint("(" + str(lamda_2) + ",\t" + str(metVal_listOfDict[lambda_2_indx][met]) + str(")"))

#-------main-------
lamda_vals = [0.0, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dataDirRoot = "../data"
args = parse_args()
V, A, instanceSuffix = argPreprocess(args, dataDirRoot)
dataset = args.dataset

resultLogFileName = "allLambdaResults_" + dataset + instanceSuffix
if (args.num_prio_lvl_EVALONLY is not None):
    resultLogFileName += "_evalOnPrioLvls" + str(args.num_prio_lvl_EVALONLY)
resultLogFileName += ".txt"

resultDirRoot = os.path.join("../results/", dataset)
schedulesDir = os.path.join(resultDirRoot, "schedules")
perfDir = os.path.join(resultDirRoot, "performance")
if not os.path.exists(perfDir):
    os.makedirs(perfDir)

resultLogPath = os.path.join(perfDir, resultLogFileName)
set_resultLogPath(resultLogPath)
f = open(resultLogPath, "w")
f.close()

if (args.num_prio_lvl is not None):
    V_partitioned_list, V_partitioned_indxList = partitionV_toPrioLevels(V, args.num_prio_lvl)
else:
    V_partitioned_list, V_partitioned_indxList = None, None

IAM_baseline(dataset, V, A, schedulesDir)
print_baseline_results(V, A, dataset, schedulesDir, instanceSuffix, args.num_prio_lvl, V_partitioned_list, V_partitioned_indxList)
print_optimization_results(V, A, dataset, schedulesDir, instanceSuffix)