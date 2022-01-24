from __future__ import print_function
import pickle,gzip,time,sys,os
import numpy as np
import argparse
from enum import Enum
import pprint
import collections

resultLogPath = None

def set_resultLogPath(myPath):
    global resultLogPath
    resultLogPath = myPath

class OPTZ_METHOD(Enum):
    ilp = 1
    lp_rptdRnd = 2
    lp_clustRptdRnd = 3

def modifiedPrint(thingToPrint, prettyPrint = False):
    global resultLogPath
    if not prettyPrint:
        print (thingToPrint)
    else:
        pprint.pprint(thingToPrint)
    if (resultLogPath is not None):
        with open(resultLogPath,'a') as f:       
            if not prettyPrint:
                print(thingToPrint, file = f)
            else:
                pprint.pprint(thingToPrint, f)
    
def prettyFormat(d, delim = "\n", keyLength = 15):
    strToReturn = ""
    dOrdered = collections.OrderedDict(sorted(d.items())) #sorting by keys
    for key, value in dOrdered.items():
        if isinstance(value, dict):
            value = prettyFormat(value, delim = ", ", keyLength = 0) #recursive call
        strToReturn += (key.ljust(keyLength) + " : " + str(value) + delim)
    if (delim != "\n"):
        strToReturn = "{" + strToReturn.strip().strip(",") + "}"
    return strToReturn

def partitionV_toPrioLevels(V, num_prio_lvl):
    # V is Interest Matrix ==> has Participants along rows & Talks along Columns
    modifiedPrint ("\n------- Partitioning V starts -------\n")
    modifiedPrint ("Example: For 3 prio-levels, V_partitioned_list contains [[Top Prio], [Mid Prio], [Low Prio]]")
    interestPerTalk = V.sum(axis=0) #column-wise sum
    Vindx_descByInterest = interestPerTalk.argsort()[::-1]
    modifiedPrint ("Total interest of each talk: {}".format(interestPerTalk))
    modifiedPrint ("Talk indices, descending by total interest: {}".format(Vindx_descByInterest))
    V_partitioned_list = []
    V_partitioned_indxList = []
    numTalksInEachPrioLvl = np.ceil(V.shape[1]/num_prio_lvl).astype('int') #equal partitions
    modifiedPrint ("numTalksInEachPrioLvl: {}".format(numTalksInEachPrioLvl))
    for prio in range(num_prio_lvl):
        strtIndx = prio * numTalksInEachPrioLvl
        endIndx = strtIndx + numTalksInEachPrioLvl if (prio != num_prio_lvl - 1) else Vindx_descByInterest.shape[0]
        V_indices_toChoose = Vindx_descByInterest[strtIndx : endIndx].tolist()
        V_partitioned_indxList.append(V_indices_toChoose)
        modifiedPrint ("--------------")
        modifiedPrint ("Talk indices, in priority-partition #{}: {}".format(prio + 1, V_indices_toChoose))
        V_partitioned = np.take(V, V_indices_toChoose, axis = 1) #choose columns whose indx is present in the list
        modifiedPrint ("Shape of Talk-matrix, for priority-partition #{}: {}".format(prio + 1, V_partitioned.shape))
        V_partitioned_list.append(V_partitioned)
    modifiedPrint ("\n------- Partitioning V ends -------\n") 
    # V_partitioned_indxList contains corresponding indices
    return V_partitioned_list, V_partitioned_indxList

def IAM_baseline(dataset, V, A, resultDir):
    modifiedPrint("============================== " + dataset)
    m=V.shape[0]
    n=V.shape[1]
    l=A.shape[1]
    X=np.zeros((n,l))
    sorted_A=np.argsort(A.sum(axis=0))
    sorted_V=np.argsort(V.sum(axis=0))
    for i in range(n):
        t=sorted_V[-1-i]
        s=sorted_A[-1-i]
        X[t,s]=1.0
    np.savetxt(resultDir + "/" + dataset + "_IAM.csv", X, delimiter=",")
    modifiedPrint(dataset + ", numTalks = " + str(n) + ", numSlots = " + str(l))
   
def giniCoeff(x):
    # Mean absolute difference
    meanAbsDiff = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    relMAD = meanAbsDiff/np.mean(x)
    # Gini coefficient
    giniCoeff = 0.5 * relMAD
    return giniCoeff

def metrics(V, A, X, num_prio_lvl = None, V_partitioned_list = None, V_partitioned_indxList = None):
    #set num_prio_lvl = None when there's no priority partitioning
    mets={}
    # m=number of participants
    m=V.shape[0]
    # n=number of talks
    n=V.shape[1]
    # l=number of slots
    l=A.shape[1]

    # ---- PARTICIPANT SIDE METRICS >>>>

    # ICGs of participants
    ICG=np.zeros(m)
    for p in range(m): #for each participant
        Interests=sorted(V[p,],reverse=True)
        Availabilities=sorted(A[p,],reverse=True)
        for t in range(n):
            ICG[p]+=(Interests[t]*Availabilities[t])

    if (num_prio_lvl is None):
        NCGs = np.diag(np.matmul(V,np.matmul(A,X.T).T))/ICG
    else:
        # global-level metric for participant (not prio-level wise)
        # gain is considered only for top-1 out of all slots
        # that is, participants attend talks only once
        CGs = np.zeros(m)
        partcpPreferredSlotsForTalks = np.full((m,n), None) # partcpPreferredSlotsForTalks[p,t] = best slot num
        for partcp in range(m):
            # each talk being offered in one/more slots
            # for each talk, consider gain only from that slot that gives highest gain for partcp
            for tlk in range(n):
                possib_slotGains = np.multiply(A[partcp,:], X[tlk,:]) #element-wise multiplication
                maxGain = V[partcp,tlk] * np.amax(possib_slotGains)
                if (np.all(possib_slotGains == 0)):
                    # if product of all A and X are 0, choose any slot where the talk is scheduled
                    partcpPreferredSlotsForTalks[partcp,tlk] = np.nonzero(X[tlk,:])[0][0]
                    assert X[tlk, partcpPreferredSlotsForTalks[partcp,tlk]] != 0
                    assert A[partcp, partcpPreferredSlotsForTalks[partcp,tlk]] == 0
                else:
                    # else choose slot that's most beneficial for partcp to attend talk tlk
                    partcpPreferredSlotsForTalks[partcp,tlk] = np.argmax(possib_slotGains) # to be used in speaker metric
                CGs[partcp] += maxGain
        NCGs = CGs / ICG
        # print ("CGs" + str(CGs.tolist()))
        # print ("ICG" + str(ICG.tolist()))
        # print ("min(NCGs)", min(NCGs))
        # print ("max(NCGs)", max(NCGs))
        # modifiedPrint ("partcpPreferredSlotsForTalks")
        # modifiedPrint (partcpPreferredSlotsForTalks)
        # for vvv in range(partcpPreferredSlotsForTalks.shape[1]):
        #      modifiedPrint (np.unique(partcpPreferredSlotsForTalks[:,vvv]))

    #modifiedPrint ("Individual Participant Satisfaction:" + str(np.sort(NCGs)))

    MeanNCG = "{:.3f}".format(np.mean(NCGs))
    PFairness = "{:.3f}".format(np.max(NCGs)-np.min(NCGs))
    PGini = "{:.3f}".format(giniCoeff(NCGs))

    # <<<< PARTICIPANT SIDE METRICS ----

    # ---- SPEAKER SIDE METRICS >>>>

    # --- IECs of speakers ---
    # IEC calculation is as per full-scale scheduling
    IEC=np.zeros(n)
    for t in range(n): #for each talk
        Interests=V[:,t] #for a particular talk
        ECs=np.matmul(Interests.T,A)
        IEC[t]=max(ECs)

    if (num_prio_lvl is None):
        NECs = np.diag(np.matmul(V.T,np.matmul(A,X.T)))/IEC
        MeanNEC = "{:.3f}".format(np.mean(NECs))
        SFairness = "{:.3f}".format(np.max(NECs)-np.min(NECs))
        SGini = "{:.3f}".format(giniCoeff(NECs))
        #modifiedPrint ("Individual Speaker Satisfaction:" + str(np.sort(NECs)))
    else:
        # local-level metric for participant (prio-level wise)
        IEC_partitioned = {}
        NECs_partitioned = {}
        MeanNEC_partitioned = {}
        SFairness_partitioned = {}
        SGini_partitioned = {}

        for prio in range(num_prio_lvl):
            IEC_partitioned["P{}".format(prio+1)] = np.take(IEC, V_partitioned_indxList[prio])
            V_partitioned = V_partitioned_list[prio]
            X_partitioned = np.take(X, V_partitioned_indxList[prio], axis = 0) #row-wise selection, bcos X is (#talks)*(#slots)
            # gain is considered only for top-1 out of all slots
            # that is, participants attend talks only once
            ECs_partitioned = np.zeros(V_partitioned.shape[1])
            for tlk in range(V_partitioned.shape[1]):
                # each talk being offered in one/more slots
                # for each talk, consider gain only from that slot that gives highest gain for partcp
                for partcp in range(m):
                    partcpPreferredSlotsForTalks_partitioned = np.take(partcpPreferredSlotsForTalks, 
                                                       V_partitioned_indxList[prio], axis = 1) #col-wise selection
                    preferredSlotForTalk = int(partcpPreferredSlotsForTalks_partitioned[partcp,tlk])
                    maxGain = V_partitioned[partcp,tlk] * A[partcp, preferredSlotForTalk]
                    ECs_partitioned[tlk] += maxGain
            NECs_partitioned["P{}".format(prio+1)] = ECs_partitioned / IEC_partitioned["P{}".format(prio+1)]
            #print ("ECs_partitioned", ECs_partitioned)
            #print ("IEC_partitioned", IEC_partitioned)

            MeanNEC_partitioned["P{}".format(prio+1)] = "{:.3f}".format(np.mean(NECs_partitioned["P{}".format(prio+1)]))
            SFairness_partitioned["P{}".format(prio+1)] = "{:.3f}".format(np.max(NECs_partitioned["P{}".format(prio+1)]) - \
                                                          np.min(NECs_partitioned["P{}".format(prio+1)]))
            SGini_partitioned["P{}".format(prio+1)] = "{:.3f}".format(giniCoeff(NECs_partitioned["P{}".format(prio+1)]))
 
    # <<<< SPEAKER SIDE METRICS ----

    # ---- SOCIAL WELFARE METRIC >>>>

    if (num_prio_lvl is None):
        TEP = np.trace(np.matmul(V,np.matmul(A,X.T).T))
    else:
        # TEP = sum of cumulative gains of all the participants, considering they will attend only once
        TEP = np.sum(CGs)

    # <<<< SOCIAL WELFARE METRIC ----
    
    mets["NCG_mean"] = MeanNCG
    mets["NCG_max-NCG_min"] = PFairness
    mets["NCG_Gini"] = PGini
    mets["NEC_mean"] = MeanNEC if (num_prio_lvl is None) else MeanNEC_partitioned
    mets["NEC_max-NEC_min"] = SFairness if (num_prio_lvl is None) else SFairness_partitioned
    mets["NEC_Gini"] = SGini if (num_prio_lvl is None) else SGini_partitioned
    mets["TEP"] = TEP
    return (mets)

def print_baseline_results(V, A, dataset, resultDir, instanceSuffix, num_prio_lvl, V_partitioned_list, V_partitioned_indxList):
    modifiedPrint("============================== baseline_results")
    # Social Welfare Maximization
    X=np.loadtxt(resultDir + "/" + dataset + "_1.0_0.0_0.0" + instanceSuffix + ".csv", delimiter=",")
    modifiedPrint("------- SWM Metrics:\n" + prettyFormat(metrics(V, A, X, num_prio_lvl, V_partitioned_list, V_partitioned_indxList)))
    # Participant Fairness Maximization
    X=np.loadtxt(resultDir + "/" + dataset + "_0.0_1.0_0.0" + instanceSuffix + ".csv", delimiter=",")
    modifiedPrint("------- PFair Metrics:\n" + prettyFormat(metrics(V, A, X, num_prio_lvl, V_partitioned_list, V_partitioned_indxList)))
    # Speaker Fairness Maximization
    X=np.loadtxt(resultDir + "/" + dataset + "_0.0_0.0_1.0" + instanceSuffix + ".csv", delimiter=",")
    modifiedPrint("------- SFair Metrics:\n" + prettyFormat(metrics(V, A, X, num_prio_lvl, V_partitioned_list, V_partitioned_indxList)))
    # Interest-Availability Matching
    X=np.loadtxt(resultDir + "/" + dataset + "_IAM.csv",delimiter=",")
    modifiedPrint("------- IAM Metrics:\n" + prettyFormat(metrics(V, A, X, num_prio_lvl, V_partitioned_list, V_partitioned_indxList)))

def argPreprocess(args, dataDir):
    optzMethod = OPTZ_METHOD(args.confType)    

    # ---- loading data ----
    V_path = dataDir + "/" + args.dataset + "_V.csv"
    A_path = dataDir + "/" + args.dataset + "_A.csv"
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
        instanceSuffix = "_ILP_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
    elif (optzMethod == OPTZ_METHOD.lp_rptdRnd):
        args.n_clusters = mOrig
        args.clusterFrac = None
        instanceSuffix = "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
    elif (optzMethod == OPTZ_METHOD.lp_clustRptdRnd):
        if (args.clusterFrac is not None):
            if (args.clusterFrac <= 0) or (args.clusterFrac > 1):
                modifiedPrint ("clusterFrac value is either non-positive or exceeds 1. Setting it to 1.0")
                args.clusterFrac = 1.0 #no clustering
            args.n_clusters = int(args.clusterFrac * mOrig) if args.clusterFrac <= 1 else V.shape[0]
            if (args.n_clusters == 0):
                args.n_clusters = 1 #at least one cluster
            instanceSuffix = "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)       
        elif (args.n_clusters is not None):
            if (args.n_clusters <= 0) or (args.n_clusters > mOrig):
                args.n_clusters = mOrig
            instanceSuffix = "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
        else:
            args.n_clusters = mOrig
            instanceSuffix = "_CN" + str(args.n_clusters) + "_FB" + str(args.fairBoundIndx)
    if (args.num_prio_lvl_EVALONLY is None) and (args.num_prio_lvl is not None):
        instanceSuffix += "_PrioLvls" + str(args.num_prio_lvl)
    else:
        instanceSuffix += "_PrioLvls" + str(1)
    return (V, A, instanceSuffix)