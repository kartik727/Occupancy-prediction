def headerData (hS): # Takes String of header and converts to different integers
    year = int(hS[1:5])
    month = int(hS[6:8])
    day = int(hS[9:11])
    
    num_att = 0
    
    return day, month, year, num_att    

def readAttendance (sD, startTime): # Gives in-time and out-time relative to class start
    if sD[0]!="P":
        return False, False, 0, 0
        
    sDs = sD.split(" ")
    
    inS = list(map(int, sDs[1].split(":")))
    outS = list(map(int, sDs[3].split(":")))
    
    if (len(inS) < 2 or len(outS) < 2):
        return False, True, 0, 0
    
    inTime = (inS[0] - startTime[0])*60 + (inS[1] - startTime[1])
    outTime = (outS[0] - startTime[0])*60 + (outS[1] - startTime[1])

    if inTime > 60:
        return False, True, 0, 0
    
    if(outTime - inTime) < 0:
        return False, True, 0, 0
    
    return True, True, inTime, outTime # isPresent, persentMarked, inTime, outTime