import numpy as np
import pickle,gzip

def participants():
    tzs=[(-5,85),(-6,85),(-7,85),(-8,85),(-10,85),(2,71),(-2,16),(-3,16),(-4,16),(-5,16),(2,53),(9,49),(-2.5,8),(-3,8),(-5,8),(-6,8),(-7,6),(2,33),(5.5,32),(1,32),(3,23),(1,22),(2,20),(2,19),(2,18),(7,3),(8,3),(10.5,4),(12,4),(8,14),(2,14),(1,15),(2,15),(3,13),(2,12),(2,13),(9,12),(2,3),(4,1),(7,1),(8,1),(9,1),(10,1),(11,1),(12,2),(2,10),(3,9),(8,7),(8,7),(3,7),(-3,7),(2,5),(8,4),(2,3),(3,3),(2,3),(-5,3),(2,2),(2,2),(2,2),(1,2),(1,2),(1,2)]
    part_tz=[]
    for x in tzs:
        tz=x[0]
        c=x[1]
        for i in range(c):
            part_tz.append(tz)
    fo=gzip.open("../../data/recsys_tzs.pkl.gz","wb")
    pickle.dump(part_tz,fo,-1)
    fo.close()
    return part_tz

def availability():
    part_tz=participants()
    m=len(part_tz)
    # 1 day= 24 hours =48 half hour slots (only 26 slots will be used for 26 papers accepted in recsys17)
    l=48
    A=np.zeros((m,l))

    for p in range(m):
        for s in range(l):
            tz=part_tz[p]
            day_start=2*(9-tz)
            if day_start<0:
                day_start+=l
            tm=day_start
            for x in range(16):
                tm+=1
                if tm>=l:
                    tm-=l
                A[p,int(tm)]=1.0
    print(len(part_tz),np.sum(A))
    np.savetxt("../../data/recsys_A.csv",A,delimiter=",")

availability()


def interests():
    part_tz=participants()
    m=len(part_tz)
    # 26 papers accepted in recsys17
    n=26 
    cites=[69,12,10,11,22,11,9,46,12,8,21,167,34,10,22,10,208,11,26,20,13,109,11,154,124,28]
    V=np.zeros((m,n))
    max_cite=max(cites)
    for t in range(n):
        prob1=np.exp(((cites[t]+0.0)/max_cite)-1)
        ran=np.random.random(m)
        for p in range(m):
            if ran[p]<=prob1:
                V[p,t]=1

    #np.savetxt("../../data/recsys_V.csv",V,delimiter=",")
   

#interests()

#=====================================LP: 24 papers filtered 24 hourly slots==========================================

def availability_LP():
    part_tz=participants()
    m=len(part_tz)
    # 1 day= 24 hours =24 one-hour slots (only 24 papers will be scheduled out of 26 papers accepted in recsys17)
    l=24
    A=np.zeros((m,l))

    for p in range(m):
        for s in range(l):
            tz=part_tz[p]
            day_start=(8-tz)
            if day_start<0:
                day_start+=l
            tm=day_start
            for x in range(8):
                tm+=1
                if tm>=l:
                    tm-=l
                A[p,int(tm)]=1.0
    print(len(part_tz),np.sum(A))
    np.savetxt("../../data/recsys_LP_A.csv",A,delimiter=",")

availability_LP()

def interests_LP():
    part_tz=participants()
    m=len(part_tz)
    # 26 papers accepted in recsys17
    # we will schedule for only 24 most interesting papers 
    n=24 
    cites=sorted([69,12,10,11,22,11,9,46,12,8,21,167,34,10,22,10,208,11,26,20,13,109,11,154,124,28],reverse=True)[:-2]
    V=np.zeros((m,n))
    max_cite=max(cites)
    for t in range(n):
        prob1=np.exp(((cites[t]+0.0)/max_cite)-1)
        ran=np.random.random(m)
        for p in range(m):
            if ran[p]<=prob1:
                V[p,t]=1

    #np.savetxt("../../data/recsys_LP_V.csv",V,delimiter=",")
   

#interests_LP()

#=================================100p sampled=============================================
def availability100():
    part_tz_all=participants()
    part_tz=np.random.choice(part_tz_all,100)
    m=len(part_tz)
    # 1 day= 24 hours =48 half hour slots (only 26 slots will be used for 26 papers accepted in recsys17)
    l=48
    A=np.zeros((m,l))

    for p in range(m):
        for s in range(l):
            tz=part_tz[p]
            day_start=2*(9-tz)
            if day_start<0:
                day_start+=l
            tm=day_start
            for x in range(16):
                tm+=x
                if tm>=l:
                    tm-=l
                A[p,int(tm)]=1.0
    print(len(part_tz),np.sum(A))
    #np.savetxt("../../data/recsys_A.csv",A,delimiter=",")

#availability100()

def interests100():
    part_tz=participants()
    m=len(part_tz)
    # 26 papers accepted in recsys17
    n=26 
    cites=[69,12,10,11,22,11,9,46,12,8,21,167,34,10,22,10,208,11,26,20,13,109,11,154,124,28]
    V=np.zeros((m,n))
    max_cite=max(cites)
    for t in range(n):
        prob1=np.exp((cites[t]/max_cite)-1)
        ran=np.random.random(m)
        for p in range(m):
            if ran[p]<=prob1:
                V[p,t]=1

    np.savetxt("../../data/recsys100_V.csv",V[:100,],delimiter=",")


#interests100()
