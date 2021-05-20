def check_cover( arr1 , arr2 ):
    next_idx = 0   
    cover = []   
    for arr1_idx in  range(len(arr1)):      
        item1 = arr1[arr1_idx]
        
        for arr2_idx in range(next_idx , len(arr2)):             
            item2 =  arr2[arr2_idx]         
            if item1[1] <= item2[0] :
                continue                  
            if item1[0] >= item2[1]:
                next_idx = arr2_idx+1
                continue                   
            cover.append(  { 'idx':[ arr1_idx , arr2_idx]  ,  'value':[ item1 , item2]})
    return cover 

def in_arr(item , arr):
    for _item in arr:
        if item[1] < _item[0] or item[0] > _item[1]:
            continue
        if item[0] >= _item[0] and item[1] <= _item[1]:
            return True    
    return False



# 丟入 s1 和  s2 的時間片段 回傳修正的 s1 和 s2
def time_fixer(s1,s2):

    s1_time ,s2_time = [] , []
    new_s1 , new_s2 = [] , []
    
    # record s1 , s2 , all time point
    for item in s1:
        s1_time.append(item[0])
        s1_time.append(item[1])
        
    for item in s2:
        s2_time.append(item[0])
        s2_time.append(item[1])  
        
    total_time = sorted(s1_time + s2_time)

    # for each loop pick two time point
    for i in range(0,len(total_time),2):

        if total_time[i] == total_time[i+1]:
            continue
        # select no overlap time
        new_item = [total_time[i] , total_time[i+1]]   
        
        #print(new_item)
        
        # decide this segment  belongs to who 
        if in_arr(new_item , s1) :
            new_s1.append(new_item)
        else:
            new_s2.append(new_item)
        
    return new_s1 , new_s2
    