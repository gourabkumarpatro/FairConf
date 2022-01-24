from eval_tools import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluating Virtual Conference Scheduling')
    parser.add_argument('--dataset', dest='dataset', type=str,
                      help='name of dataset/conference')
    parser.add_argument('--lam', dest='lam', type=float,
                      help='determines weight of efficiency')
    parser.add_argument('--lam1', dest='lam1', type=float,
                      help='determines weight of participant fairness')
    parser.add_argument('--lam2', dest='lam2', type=float,
                      help='determines weight of speaker fairness')
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
    parser.add_argument('--scheduleName_overrided', dest='scheduleName_overrided', type=str, default=None,
                      help='name of schedule .csv file (dont include extension). If this is provided, evaluates this schedule only')
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
    global args, V_partitioned_list, V_partitioned_indxList, lam, lam1, lam2
    if (args.scheduleName_overrided is None):
        instanceName = dataset + "_{}_{}_{}".format(lam, lam1, lam2) + instanceSuffix + ".csv"
    else:
        instanceName = args.scheduleName_overrided + ".csv"
    modifiedPrint("============================== Results of instance \'" + instanceName + "\'")
    X = np.loadtxt(resultDir + "/" + instanceName, delimiter=",")
    modifiedPrint(prettyFormat(metrics(V, A, X, args.num_prio_lvl, V_partitioned_list, V_partitioned_indxList)))

#-------main-------
dataDirRoot = "../data"
args = parse_args()
V, A, instanceSuffix = argPreprocess(args, dataDirRoot)
dataset, lam, lam1, lam2 = args.dataset, args.lam, args.lam1, args.lam2

# -- log file >>
if (args.scheduleName_overrided is None):
    resultLogFileName = "singleResult_" + dataset + "_{}_{}_{}".format(lam, lam1, lam2) + instanceSuffix
else:
    resultLogFileName = "singleResult_" + args.scheduleName_overrided
if (args.num_prio_lvl_EVALONLY is not None):
    resultLogFileName += "_evalOnPrioLvls" + str(args.num_prio_lvl_EVALONLY)
resultLogFileName += ".txt"
# << log file --

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