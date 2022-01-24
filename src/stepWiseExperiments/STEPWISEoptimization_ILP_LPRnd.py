from __future__ import print_function
import os
os.environ["OMP_NUM_THREADS"] = "3"
import pickle,gzip,gurobipy,time,sys
import numpy as np
import cvxpy as cp
import argparse
from sklearn.cluster import KMeans
import random
from enum import Enum
import copy
SET_LOW_VAL_TO_ZERO = False

#: Any number smaller than this will be rounded down to 0 when computing the difference between NumPy arrays of floats.
TOLERANCE = np.finfo(np.float).eps * 10.
A_V_TOLERANCE = 10**(-4)

class OPTZ_METHOD(Enum):
    ilp = 1
    lp_rptdRnd = 2
    lp_clustRptdRnd = 3

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Scheduling Virtual Conferences Fairly')
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
    parser.add_argument('--num_prio_lvl_EVALONLY', dest='num_prio_lvl_EVALONLY', type=int, default=None,
                      help='flag to calc metrics similar to prio, even if schedule is not of prioritized talks')
    #num_prio_lvl_EVALONLY flag ignored in this program

    args = parser.parse_args()
    if (args.clusterFrac is not None) and (args.n_clusters is not None):
        print ("Both clusterFrac and n_clusters cannot be supplied. Please supply only one of them.")
        sys.exit(0)
    return args

def clusterParticipants(partFeatures, n_clusters, nOrig, lOrig):
    print ("\nClustering participants:", \
             ", n_clusters =", n_clusters)
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++').fit(partFeatures)
    kmeansLabels = kmeans.labels_
    uniqueLbls, countLbls = np.unique(kmeansLabels, return_counts=True)
    lblCntDict = dict(zip(uniqueLbls, countLbls))
    cntPerCluster = np.array([lblCntDict[i] for i in range(max(kmeansLabels)+1)])
    kmeans_ClustCenters = kmeans.cluster_centers_
    V = np.zeros((n_clusters, nOrig))
    A = np.zeros((n_clusters, lOrig))
    for clusterNo in range(n_clusters):
        V[clusterNo, :] = kmeans_ClustCenters[clusterNo, : nOrig].clip(0).clip(max = 1)
        A[clusterNo, :] = kmeans_ClustCenters[clusterNo, nOrig : ].clip(0).clip(max = 1)
        
    #setting lower values of V to 0
    if (SET_LOW_VAL_TO_ZERO):
        print ("For V, count of vals below " + str(A_V_TOLERANCE) + ": " + str(np.sum(V <= A_V_TOLERANCE)) +
               ". Setting them to 0")
        V[V <= A_V_TOLERANCE] = 0
        V_zeroRows = np.where(abs(V.sum(axis=1)) <= A_V_TOLERANCE)[0]
        V_zeroColumns = np.where(abs(V.sum(axis=0)) <= A_V_TOLERANCE)[0]
        print ("For V, rows in which sum <= TOLERANCE:", V_zeroRows)
        print ("For V, columns in which sum <= TOLERANCE:", V_zeroColumns)
    
    #setting lower values of A to 0
    if (SET_LOW_VAL_TO_ZERO):
        print ("For A, count of vals below " + str(A_V_TOLERANCE) + ": " + str(np.sum(A<= A_V_TOLERANCE)) +
               ". Setting them to 0")
        A[A <= A_V_TOLERANCE] = 0
        A_zeroRows = np.where(abs(A.sum(axis=1)) <= A_V_TOLERANCE)[0]
        A_zeroColumns = np.where(abs(A.sum(axis=0)) <= A_V_TOLERANCE)[0]
        print ("For A, rows in which sum <= TOLERANCE:", A_zeroRows)
        print ("For A, columns in which sum <= TOLERANCE:", A_zeroColumns)
    print ("Clustering Done!\n") 
    return V, A, cntPerCluster

def conference_schedule(V, A, lamda, lamda_1, lamda_2, clusterCnt, optzMethod):
    global args
    # JOINT OPTIMIZATION FOR FAIR CONFERENCE SCHEDULE
    #size of V is (m*n), size of A is (n*l)
    # m=number of participants
    m=V.shape[0]

    # n=number of talks
    n=V.shape[1]

    # l=number of slots
    l=A.shape[1]
    
    V_weighted = V * clusterCnt[:,np.newaxis] #mult ith row of V by ith val of clusterCnt
    #V_weighted used in social welfare, speaker fairness, and IEC term

    # input validity check
    if V.shape[0]!=A.shape[0]:
        print("Invalid input: m value does not match")
        return(0)
    if n>l:
        print("Invalid input: more talks and less slots")
        return(0)
    if (V>1).sum().sum() or (V<0).sum().sum():
        print("Invalid input: invalid values in V")
        return(0)
    if (A>1).sum().sum() or (A<0).sum().sum():
        print("Invalid input: invalid values in A")
        return(0)
    
    # ICGs for participants
    ICG=np.zeros(m)
    for p in range(m):
        Interests=sorted(V[p,],reverse=True)
        Availabilities=sorted(A[p,],reverse=True)
        for t in range(n):
            ICG[p]+=(Interests[t]*Availabilities[t])

    # IECs for speakers
    IEC=np.zeros(n)
    overall_A=A.sum(axis=0)
    for t in range(n):
        Interests=V_weighted[:,t]
        ECs=np.matmul(Interests.T,A)
        IEC[t]=max(ECs)

    #==================Code for ILP/LP starts here
    start=time.time()

    if (optzMethod == OPTZ_METHOD.ilp):
        assert (V == V_weighted).all()
        X=cp.Variable((n,l), boolean=True)
        print ("Solving ILP:", "Participants:", m, "; Talks:", n, "; Slots:", l)
    else:
        X=cp.Variable((n,l))
        print ("Solving LP:", "Participants:", m, "; Talks:", n, "; Slots:", l)

    #Constraints
    constraints=[]

    #Talk-side Constraint: Each talk must be assigned to exactly one slot
    #Row-sums must be 1 
    constraints.append(cp.sum(X,axis=1)==1)

    #Slot-side Constraint: Each slot can be assigned to at most one talk (LP Trial)
    constraints.append(cp.sum(X,axis=0) <= 1)

    if not (optzMethod == OPTZ_METHOD.ilp):
        # LP-specific non-negativity constraint
        constraints.append(X>=0)

    #LP Objective Function
    if (args.fairBoundIndx == 1 or m==1 or n==1):
        obj_fn=cp.Maximize( lamda*cp.trace(cp.matmul(V_weighted,cp.matmul(A,X.T).T))/(m*n) +\
                           lamda_1*( cp.min(cp.diag(cp.matmul(V,cp.matmul(A,X.T).T))/ICG) -\
                                    cp.max(cp.diag(cp.matmul(V,cp.matmul(A,X.T).T))/ICG) ) +\
                           lamda_2*( cp.min(cp.diag(cp.matmul(V_weighted.T,cp.matmul(A,X.T)))/IEC) -\
                                    cp.max(cp.diag(cp.matmul(V_weighted.T,cp.matmul(A,X.T)))/IEC) ) )
    else:
        if (m > args.fairBoundIndx):
            fBndIndxICG = args.fairBoundIndx
        else:
            fBndIndxICG = m-1
        if (n > args.fairBoundIndx):
            fBndIndxIEC = args.fairBoundIndx
        else:
            fBndIndxIEC = n-1 
        obj_fn=cp.Maximize( lamda*cp.trace(cp.matmul(V_weighted,cp.matmul(A,X.T).T))/(m*n) +\
                            lamda_1*(
                                    (cp.sum_smallest(cp.diag(cp.matmul(V,cp.matmul(A,X.T).T))/ICG,
                                                                                fBndIndxICG) -
                                    cp.sum_smallest(cp.diag(cp.matmul(V,cp.matmul(A,X.T).T))/ICG,
                                                                                fBndIndxICG - 1)) -\
                                    (cp.sum_largest(cp.diag(cp.matmul(V,cp.matmul(A,X.T).T))/ICG,
                                                                                fBndIndxICG) -
                                    cp.sum_largest(cp.diag(cp.matmul(V,cp.matmul(A,X.T).T))/ICG,
                                                                                fBndIndxICG - 1))
                                    ) +\
                            lamda_2*(
                                   (cp.sum_smallest(cp.diag(cp.matmul(V_weighted.T,cp.matmul(A,X.T)))/IEC,
                                                                                fBndIndxIEC) -
                                    cp.sum_smallest(cp.diag(cp.matmul(V_weighted.T,cp.matmul(A,X.T)))/IEC,
                                                                                fBndIndxIEC - 1)) -\
                                   (cp.sum_largest(cp.diag(cp.matmul(V_weighted.T,cp.matmul(A,X.T)))/IEC,
                                                                                fBndIndxIEC) -
                                    cp.sum_largest(cp.diag(cp.matmul(V_weighted.T,cp.matmul(A,X.T)))/IEC,
                                                                                fBndIndxIEC - 1))
                                    )
                            ) 

    #Problem Definition and Solve
    print ("Problem Definition starts.")
    prob=cp.Problem(obj_fn,constraints)
    print ("Problem solving starts.")
    if (args.fairBoundIndx == 1):
        prob.solve(solver=cp.GUROBI, verbose = False, NodefileStart = 0.5, Threads = 3)
    else:
        prob.solve(solver=cp.GUROBI, verbose = False, NodefileStart = 0.5, Threads = 3) #gp = True
    print ("Problem solving ends.")
    end=time.time()

    #Returning the solution
    print ("Shape of X:", X.shape)
    print ("------------------")
    return X.value
    
def LP_rounding(X, V, A, talkSlotDict, talkIndxToNumMap, slotIndxToNumMap, lamda, lamda_1, lamda_2):
    # ICGs and IEGs
    # m=number of participants
    m=V.shape[0]

    # n=number of talks
    n=V.shape[1]

    # l=number of slots
    l=A.shape[1]

    print ("Rounding LP Solution:", "Participants:", m, "; Talks:", n, "; Slots:", l)
    print ("Shape of X:", X.shape)

    #mapMatrix = np.zeros(X.shape, dtype = int) # (i,j) = 1 indicates talk of ith row mapped to slot at jth column
    talk_flagList = [False for i in range(n)] # True indicates talk at that index is scheduled
    slot_flagList = [False for i in range(l)] # True indicates slot at that index is scheduled
    cnt = 1

    while (not np.all(abs(X) < TOLERANCE)):
        i, j = np.unravel_index(X.argmax(), X.shape)
        origTalkNum = talkIndxToNumMap[i]
        origSlotNum = slotIndxToNumMap[j]
        assert (origTalkNum not in talkSlotDict)
        talkSlotDict[origTalkNum] = origSlotNum #(origTalkNum)th talk assigned to (origSlotNum)th slot
        print ('%02d' % cnt + ")", "maxVal =", "%.7f" % X[i, j], "at X[" + '%02d' % i + ", " +\
                 '%02d' % j + "] ;",) 
        if (origSlotNum is not None):
            print ("\tTalk " + '%02d' % origTalkNum + " mapped to Slot " + '%02d' % origSlotNum)
        else:
            print ("\tTalk " + '%02d' % origTalkNum + " mapped to Slot " + str(origSlotNum))

        X[i, :] = 0
        X[:, j] = 0
        talk_flagList[i] = True
        slot_flagList[j] = True
        cnt += 1

    talkIndx = 0
    NEWtalkIndxToNumMap = {}
    for oldTalkIndx in range(len(talk_flagList)):
        if (talk_flagList[oldTalkIndx] == False): #still un-scheduled talk, will be present in next talk list
            NEWtalkIndxToNumMap[talkIndx] = talkIndxToNumMap[oldTalkIndx]
            talkIndx += 1 #gets incremented for each un-scheduled talk

    slotIndx = 0
    NEWslotIndxToNumMap = {}
    for oldSlotIndx in range(len(slot_flagList)):
        if (slot_flagList[oldSlotIndx] == False): #still free slot, will be present in next slot list
            NEWslotIndxToNumMap[slotIndx] = slotIndxToNumMap[oldSlotIndx]
            slotIndx += 1 #gets incremented for each free slot


    print ("------------------")
    doneFlag = all(talk_flagList)
    unscheduledTalkIndices = np.where(np.array(talk_flagList) == False)[0]
    freeSlotIndices = np.where(np.array(slot_flagList) == False)[0]
    newV = V[:, unscheduledTalkIndices]
    newA = A[:, freeSlotIndices]
    
    return (doneFlag, talkSlotDict, NEWtalkIndxToNumMap, NEWslotIndxToNumMap, newV, newA)

def partitionV_toPrioLevels(V, num_prio_lvl):
    # V is Interest Matrix ==> has Participants along rows & Talks along Columns
    print ("\n------- Partitioning V starts -------\n")
    print ("Example: For 3 prio-levels, V_partitioned_list contains [[Top Prio], [Mid Prio], [Low Prio]]")
    interestPerTalk = V.sum(axis=0) #column-wise sum
    Vindx_descByInterest = interestPerTalk.argsort()[::-1]
    print ("Total interest of each talk:", interestPerTalk)
    print ("Talk indices, descending by total interest:", Vindx_descByInterest)
    V_partitioned_list = []
    V_partitioned_indxList = []
    numTalksInEachPrioLvl = np.ceil(V.shape[1]/num_prio_lvl).astype('int') #equal partitions
    print ("numTalksInEachPrioLvl:", numTalksInEachPrioLvl)
    for prio in range(num_prio_lvl):
        strtIndx = prio * numTalksInEachPrioLvl
        endIndx = strtIndx + numTalksInEachPrioLvl if (prio != num_prio_lvl - 1) else Vindx_descByInterest.shape[0]
        V_indices_toChoose = Vindx_descByInterest[strtIndx : endIndx].tolist()
        V_partitioned_indxList.append(V_indices_toChoose)
        print ("--------------")
        print ("Talk indices, in priority-partition #{}:".format(prio + 1), V_indices_toChoose)
        V_partitioned = np.take(V, V_indices_toChoose, axis = 1) #choose columns whose indx is present in the list
        print ("Shape of Talk-matrix, for priority-partition #{}:".format(prio + 1), V_partitioned.shape)
        V_partitioned_list.append(V_partitioned)
    print ("\n------- Partitioning V ends -------\n") 
    # V_partitioned_indxList contains corresponding indices
    return V_partitioned_list, V_partitioned_indxList

def performScheduling(num_prio_lvl, lamda, lamda_1, lamda_2,\
                       V, A, cntPerCluster, instanceName, nOrig, lOrig, optzMethod):
    mapMatrixDict = {}
    if (num_prio_lvl):
        V_partitioned_list, V_partitioned_indxList = partitionV_toPrioLevels(V, num_prio_lvl)
    else:
        V_partitioned_list = [V]
        num_prio_lvl = 1
        V_partitioned_indxList = [list(range(V.shape[1]))]
    # For 3 prio-levels, V_partitioned_list contains V's partition as [[TopPrio Talks], [MidPrio Talks], [LowPrio Talks]]
    # For 3 prio-levels, V_partitioned_indxList contains corresponding V indices as [[TopPrio indx], [MidPrio Indx], [LowPrio Indx]]

    # ---- conference scheduling ----
    mapMatrix = np.zeros((nOrig, lOrig), dtype = int)
    # (i,j) = 1 indicates talk of ith row mapped to slot at jth column
    firstTimeLoopEntry = True

    # ---- LP repeated rounding or ILP ----
    assert (len(V_partitioned_list) == num_prio_lvl)
    noSlotsLeftFLAG = False
    for round in range(num_prio_lvl):
        # For 3 prios: in 1st rnd all_3_prios, in 2nd rnd 1st_&_2nd_prios, in 3rd rnd 1st_prio
        # So, 1st prio scheduled thrice, 2nd prio scheduled twice, 3rd prio scheduled once
        V_partitioned_list_rnd = V_partitioned_list[:len(V_partitioned_list)-round]
        V_partitioned_indxList_rnd = V_partitioned_indxList[:len(V_partitioned_indxList)-round]

        ## ---- Experiment for 1 prio: 2 rounds of full-scale scheduling (num_prio = 1) >>>>
        #V_partitioned_list_rnd = [V_partitioned_list_rnd[0], V_partitioned_list_rnd[0]]
        #V_partitioned_indxList_rnd = [V_partitioned_indxList_rnd[0], V_partitioned_indxList_rnd[0]]
        ## <<<< Experiment for 1 prio: 2 rounds of full-scale scheduling (num_prio = 1) ----

        loopVal = 1
        if round == 2:
            loopVal = 2

        for xxx in range(loopVal):
            if (xxx == 0 and loopVal == 2):
	        # ---- Experiment for 3 prios: (P1,P2,P3),(P1,P2),P1 >>>>
                V_partitioned_list_rnd = V_partitioned_list[:len(V_partitioned_list)-round]
                V_partitioned_indxList_rnd = V_partitioned_indxList[:len(V_partitioned_indxList)-round]   
                # <<<< Experiment for 3 prios: (P1,P2,P3),(P1,P2),P1 ----				
            elif (xxx == 1 and loopVal == 2):
                # ---- Experiment for 3 prios: (P1,P2,P3),(P1,P2),P3 >>>>
                V_partitioned_list_rnd = [V_partitioned_list[round]]
                V_partitioned_indxList_rnd = [V_partitioned_indxList[round]]
                # <<<< Experiment for 3 prios: (P1,P2,P3),(P1,P2),P3 ----
            if (loopVal == 2):
                V = copy.deepcopy(V_stored)
                A = copy.deepcopy(A_stored)
                talkIndxToNumMap = copy.deepcopy(talkIndxToNumMap_stored)
                slotIndxToNumMap = copy.deepcopy(slotIndxToNumMap_stored)
                scheduledAllTalksFlag = copy.deepcopy(scheduledAllTalksFlag_stored)
                talkSlotDict = copy.deepcopy(talkSlotDict_stored)
                mapMatrix = copy.deepcopy(mapMatrix_stored)
            if (noSlotsLeftFLAG):
                break
            for partnIndx in range(len(V_partitioned_indxList_rnd)):
                scheduledAllTalksFlag = False
                talkSlotDict = {}     #contains mapping of orig talk no. to orig slot no.
                talkIndxToNumMap = {} #for each iteration, maps talk index to original talk no.

                V = V_partitioned_list_rnd[partnIndx]
                for talkNo in range(V.shape[1]):
                    talkIndxToNumMap[talkNo] = V_partitioned_indxList_rnd[partnIndx][talkNo]
            
                if firstTimeLoopEntry:
                    slotIndxToNumMap = {} #for each iteration, maps slot index to original slot no.
                    for slotNo in range(A.shape[1]):
                        slotIndxToNumMap[slotNo] = slotNo
                    firstTimeLoopEntry = False
          
                if (A is None) or (A.shape[1] == 0): # no more slots left
                    noSlotsLeftFLAG = True
                    print ("No more slots left...exiting")
                    break

                print ("\n------------ ROUND {}: Scheduling priority-partition {}/{} (Starts at 1, Low no. = High Prio) ------------\n".format(round + 1,
                                                                                                partnIndx + 1, num_prio_lvl))
                if (A.shape[1] < V.shape[1]):
                    print ("No. of slots left < No. of talks ... adding extra slots with 0 availability")
                    A_tmp = copy.deepcopy(A)
                    A = np.zeros((A.shape[0], V.shape[1]))
                    A[:,:A_tmp.shape[1]] = A_tmp
                    print ("\tA.shape[1] = {}, V.shape[1] = {}".format(A_tmp.shape[1], V.shape[1]))
                    for slotIndx in range(A_tmp.shape[1], V.shape[1]):
                        assert (slotIndx not in slotIndxToNumMap)
                        # add dummy slots
                        slotIndxToNumMap[slotIndx] = None

                #print ("slotIndxToNumMap", slotIndxToNumMap)
                #print ("talkIndxToNumMap", talkIndxToNumMap)
                while (not scheduledAllTalksFlag):
                    Res = conference_schedule(V,A,lamda,lamda_1,lamda_2,cntPerCluster,optzMethod)
                    scheduledAllTalksFlag, talkSlotDict, talkIndxToNumMap, slotIndxToNumMap, V, A =\
                                        LP_rounding(Res, V, A, talkSlotDict, talkIndxToNumMap, 
                                                    slotIndxToNumMap, lamda, lamda_1, lamda_2)
                    if (optzMethod == OPTZ_METHOD.ilp):
                        # ---- ILP ----
                        assert scheduledAllTalksFlag
                    print ("A's shape now:", A.shape)
                    # A & V gets modified each time

                for talk in list(talkSlotDict.keys()):
                    slot = talkSlotDict[talk]
                    if (slot is not None):
                        mapMatrix[talk, slot] = 1
                # ---- Experiment: Only till 2nd round, 1st partn (e.g. P1,P2,P3,P1 for 3 prio) >>>>
                if (round == 1 and partnIndx == 0):
                    print ("Exiting after 2nd round, 1st partn")
                    mapMatrixDict['P1P2P3P1'] = copy.deepcopy(mapMatrix)
                # <<<< Experiment: Only till 1st round ----

            ## ---- Experiment: Only till 1st round (e.g. P1,P2,P3 for 3 prio) >>>>
            if (round == 0):
                print ("Exiting after 1st round")
                mapMatrixDict['P1P2P3'] = copy.deepcopy(mapMatrix) 
            # <<<< Experiment: Only till 1st round ----
            # ---- Experiment: Only till 2nd round (e.g. P1,P2,P3,P1,P2 for 3 prio) >>>>
            if (round == 1):
                print ("Exiting after 2nd round")
                mapMatrixDict['P1P2P3P1P2'] = copy.deepcopy(mapMatrix)
                V_stored = copy.deepcopy(V)
                A_stored = copy.deepcopy(A)
                talkIndxToNumMap_stored = copy.deepcopy(talkIndxToNumMap)
                slotIndxToNumMap_stored = copy.deepcopy(slotIndxToNumMap)
                scheduledAllTalksFlag_stored = copy.deepcopy(scheduledAllTalksFlag)
                talkSlotDict_stored = copy.deepcopy(talkSlotDict)
                mapMatrix_stored = copy.deepcopy(mapMatrix)
            # <<<< Experiment: Only till 2nd round ----
            # ---- Experiment: Only till 3rd round (e.g. P1,P2,P3,P1,P2,P1 for 3 prio) >>>>
            if (round == 2 and xxx == 0):
                print ("Exiting after 3rd round")
                mapMatrixDict['P1P2P3P1P2P1'] = copy.deepcopy(mapMatrix)
            elif (round == 2 and xxx == 1):
                print ("Exiting after 3rd round")
                mapMatrixDict['P1P2P3P1P2P3'] = copy.deepcopy(mapMatrix)
            # <<<< Experiment: Only till 3rd round ----

    return mapMatrixDict
    
def argPreprocess(args, readDir):
    # ---- parsing cmd line args ----
    optzMethod = OPTZ_METHOD(args.confType)
    print ("Dataset:", args.dataset, ", lambda:", args.lam, ", lambda_1:", args.lam1,
            ", lambda_2:", args.lam2, ", optmz_method:", optzMethod,"\n")
          
    # ---- loading data ----
    V_path = readDir + "/" + args.dataset + "_V.csv"
    A_path = readDir + "/" + args.dataset + "_A.csv"
    if (os.path.exists(V_path)):
        V = np.loadtxt(V_path, delimiter=",")
    else:
        print ("Interest matrix does not exist at path " + V_path)
        sys.exit(0)
    if (os.path.exists(A_path)):
        A = np.loadtxt(A_path, delimiter=",")
    else:
        print ("Availability matrix does not exist at path " + A_path)
        sys.exit(0)    
    assert V.shape[0] == A.shape[0] # number of participants same in both A and V
    
    # ---- refining inputs, converting all to n_clusters perspective ----
    mOrig = V.shape[0] # m=number of participants
    nOrig = V.shape[1] # n=number of talks
    lOrig = A.shape[1] # l=number of slots
    
    if (optzMethod == OPTZ_METHOD.ilp):
        args.n_clusters = mOrig
        args.clusterFrac = None
        instanceName = args.dataset + "_" + str(args.lam) + "_" + str(args.lam1) + "_" +\
                    str(args.lam2) + "_ILP_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
    elif (optzMethod == OPTZ_METHOD.lp_rptdRnd):
        args.n_clusters = mOrig
        args.clusterFrac = None
        instanceName = args.dataset + "_" + str(args.lam) + "_" + str(args.lam1) + "_" +\
                    str(args.lam2) + "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
    elif (optzMethod == OPTZ_METHOD.lp_clustRptdRnd):
        if (args.clusterFrac is not None):
            if (args.clusterFrac < 0) or (args.clusterFrac > 1):
                print ("clusterFrac value is either non-positive or exceeds 1. Setting it to 1.0")
                args.clusterFrac = 1.0 #no clustering
            args.n_clusters = int(args.clusterFrac * mOrig) if args.clusterFrac <= 1 else V.shape[0]
            if (args.n_clusters == 0):
                print ("Value of clusterFrac is too small! " +\
                        "Setting n_clusters to 1.0")
                args.n_clusters = 1 #at least one cluster
            instanceName = args.dataset + "_" + str(args.lam) + "_" + str(args.lam1) + "_" +\
                        str(args.lam2) + "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)       
        elif (args.n_clusters is not None):
            if (args.n_clusters <= 0) or (args.n_clusters > mOrig):
                print ("Value of n_clusters is either non-positive or exceeds no. of participants! " +\
                        "Setting it to no. of participants.")
                args.n_clusters = mOrig
            instanceName = args.dataset + "_" + str(args.lam) + "_" + str(args.lam1) + "_" +\
                        str(args.lam2) + "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
        else:
            args.n_clusters = mOrig
            instanceName = args.dataset + "_" + str(args.lam) + "_" + str(args.lam1) + "_" +\
                        str(args.lam2) + "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
    
    if (args.num_prio_lvl is not None):
        instanceName += "_PrioLvls" + str(args.num_prio_lvl)
    else:
        instanceName += "_PrioLvls" + str(1)

    # ---- Hereafter, args.n_clusters is definitely defined ----

    # ---- clustering data ----
    print ("OrigParticipants:", mOrig, "; OrigTalks:", nOrig, "; OrigSlots:", lOrig)
    partFeatures = np.zeros((mOrig, nOrig + lOrig), dtype = np.float64)
    partFeatures[ : , : nOrig] = V
    partFeatures[ : , nOrig : ] = A

    if (args.n_clusters < mOrig): #clustering meaningful only if (n_clusters < m)
        V, A, cntPerCluster = clusterParticipants(partFeatures, args.n_clusters, nOrig, lOrig)
        print ("Participants clustered to " + str(args.n_clusters) + " unique profiles.\n" +\
                "[#(Participants) in a cluster] -> Min: " + str(np.min(cntPerCluster)) +\
                ", Max: " + str(np.max(cntPerCluster)) + ", Mean: " + str(np.mean(cntPerCluster)))
    else:
        cntPerCluster = np.ones(mOrig) #no clustering done => each cluster contains one participant
        print ("Not clustering participants. All participants will be considered as separate profiles.")
    print ("\n")
    return (args.dataset, args.lam, args.lam1, args.lam2, V, A, cntPerCluster, instanceName, nOrig, lOrig, optzMethod)


#=========== MAIN ===========
readDirRoot = "../data"
args = parse_args()
print ("Arguments: {}".format(args))
dataset, lamda, lamda_1, lamda_2, V, A, cntPerCluster, instanceName, nOrig, lOrig, optzMethod =\
                                argPreprocess(args, readDirRoot)

writeDirRoot = os.path.join("../results/", dataset, "schedules")
         
# ---- check if already present ----
if (os.path.exists(writeDirRoot + "/" + instanceName + ".csv")):
    print ("Instance \"" + instanceName + "\" already done previously!")
    print ("\n---------------------------------------------------------------------------")
    sys.exit(0) 

mapMatrixDict = performScheduling(args.num_prio_lvl, lamda, lamda_1, lamda_2,\
                       V, A, cntPerCluster, instanceName, nOrig, lOrig, optzMethod)
print ("\nTalk-Slot Map Matrix:\n {}".format(mapMatrixDict))

## ---- UNCOMMENT FOR ADDITIONAL CHECK >>>>
#if (args.num_prio_lvl is None):  #each talk scheduled only once     
#    assert list(set(np.sum(mapMatrix, axis = 1))) == [1] #sum of all rows is 1
## <<<< UNCOMMENT FOR ADDITIONAL CHECK ----

# ---- saving the results ----
if (not os.path.exists(writeDirRoot)):
    os.makedirs(writeDirRoot)
for key in mapMatrixDict:
    thisInstName = instanceName.split("_")[0] + "_" + key +"_" + "_".join(instanceName.split("_")[1:])
    np.savetxt(writeDirRoot + "/" + thisInstName + ".csv", mapMatrixDict[key].astype(int), fmt='%i', delimiter = ",") 
    print ("Instance \"" + thisInstName + "\" saved!")
    print ("\n---------------------------------------------------------------------------")   
