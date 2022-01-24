# -*- coding: utf-8 -*-
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

from bs4 import BeautifulSoup
import numpy as np
import pickle,gzip
import copy

def participants():
    tzs=[(0, 120), (1, 650), (2, 190), (3, 50), (4, 10), (5, 50), (6, 5), (7, 15), (8, 220), (9, 130), (10, 40), (11, 10), (12, 10), (13, 15), (14, 15), (15, 20), (16, 60), (17, 440), (18, 40), (19, 210), (20, 350), (21, 40), (22, 10), (23, 22)]
    part_tz=[]
    for x in tzs:
        tz=x[0]
        c=x[1]
        for i in range(c):
            part_tz.append(tz)
    fo=gzip.open("../../data/icml_tzs.pkl.gz","wb")
    pickle.dump(part_tz,fo,-1)
    fo.close()
    return part_tz

def availability():
    # Availability values:
    # (0-2) -> 0, (2-4) -> 0, (4-6) -> 0, (6-8) -> 0.5, (8-10) -> 1, (10-12) -> 1,
    # (12,14) -> 1, (14,16) -> 1, (16-18) -> 1, (18-20) -> 1, (20,22) -> 0.5, (22-0) -> 0
    #since slots are 0.5 hrs, so 48 slots in a day. Among these, (6-22) i.e. 24 slots are important (hv non-zero val)
    #In these 24 slots, first 4 are 0.5, last 4 are 0.5, rest are 1.
    part_tz=participants()
    m=len(part_tz)
    # 1 day= 24 hours =48 half hour slots 
    l = 48
    numDays = 5
    A_oneDay=np.zeros((m,l))

    for p in range(m):
        for s in range(l):
            tz=part_tz[p]
            day_start = 2*(6-tz) - 1
            if (day_start < 0):
                day_start+=l
            tm = copy.deepcopy(day_start)
            for x in range(32):
                tm+=1
                if tm>=l:
                    tm-=l
                if x in range(0,4):
                    A_oneDay[p,int(tm)]=0.5
                elif x in range(28,32):
                    A_oneDay[p,int(tm)]=0.5
                else:
                    A_oneDay[p,int(tm)]=1.0
    A = np.tile(A_oneDay, (1, numDays)) 
    #since, no day-wise preference. Availabilty scores will repeat every 24 hours.
    print ("A's shape:", A.shape)
    print ("A's sum:", np.sum(A))
    nonZeroRows = np.where(~A.any(axis=1))[0]
    nonZeroColumns = np.where(~A.any(axis=0))[0]
    print ("For A, rows in which sum = 0:", nonZeroRows)
    print ("For A, columns in which sum = 0:", nonZeroColumns)
    if (nonZeroRows.size != 0 or nonZeroColumns.size != 0): #one non-zero row or column
        return False
    np.savetxt("../../data/icml_A.csv",A,delimiter=",")
    return True

def interests():
    part_tz = participants()
    m = len(part_tz)

    #read html file obtained as source from https://dl.acm.org/doi/proceedings/10.5555/3305381
    if sys.version[0] == '2':
        soup = BeautifulSoup(open("data/icml.html"), features="html.parser")
    else:
        soup = BeautifulSoup(open("data/icml.html", encoding="utf8"), features="html.parser")
    mydivs = soup.find_all("div", {"class": "metric"})
    #no. of cites of each paper
    cites = [int(res.text.split("Downloads")[1].replace(',', '')) for res in mydivs]
    print ("Cites:", cites)
    citesNP = np.array(cites)

    #get no. of pages for each paper
    mydivsPages = soup.find_all("div", {"class": "issue-item__detail"})
    mydivsPagesTxt = [(res.text.replace("–","-")) for res in mydivsPages]
    mydivsPagesTxt = [x.split("pp")[1].strip().split('-') for x in mydivsPagesTxt]
    numPages = [int(x[1]) - int(x[0]) + 1 for x in mydivsPagesTxt]
    print ("NumPages:", numPages)
    numPagesNP = np.array(numPages)

    citesFullLengthPPr = citesNP[numPagesNP >= 8] #choose only full-length papers i.e. papers with numPages >= 8
    print ("Cites of Full-Length Papers:\n", citesFullLengthPPr)
    n = len(citesFullLengthPPr)
    print ("No. of talks:", n)

    V=np.zeros((m,n))
    max_cite=np.max(citesFullLengthPPr)
    for t in range(n):
        prob1 = ((citesFullLengthPPr[t]+0.0)/max_cite)
        if (citesFullLengthPPr[t] == max_cite):
            print ("TalkNo. with max Cite:", t, ", Probability:", prob1)
        ran=np.random.random(m)
        for p in range(m):
            if ran[p]<=prob1:
                V[p,t]=1
    nonZeroRows = np.where(~V.any(axis=1))[0]
    nonZeroColumns = np.where(~V.any(axis=0))[0]
    print ("For V, rows in which sum = 0:", nonZeroRows)
    print ("For V, columns in which sum = 0:", nonZeroColumns)
    print ("V's shape:", V.shape)
    if (nonZeroRows.size != 0 or nonZeroColumns.size != 0): #one non-zero row or column
        return False
    np.savetxt("../../data/icml_V.csv",V,delimiter=",")
    return True

Vdone = False
while (not Vdone):    
    Vdone = interests() #not All NonZero Row/Col
    if (not Vdone):
        print ("Not all row/col non-zero in V...Regenerating")
Adone = availability()
if (not Adone):
    print ("Not all row/col non-zero in A...Exiting")

