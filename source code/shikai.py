import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
w=[1,1,1,1,1,1,1.2,3]
#def loss_x(x,array):
	# sum=0
	# w=np.random.rand(len(array))
	# w=np.sort(w)
	# w=[1,1,2,2,3,3,4,4]
	# for index,i in enumerate(array):
	# 	sum=sum+abs(x-i)/(x+i)*w[index]
	# return sum




# def zhouX_global_min(dataframe):
# 	L=[]
# 	for i in range(1,2001):
# 		tmp=dataframe.ix[i]
# 		all_loss=99999999
# 		tmp_value=-1
# 		for index,j in enumerate(tmp):
# 			tmp_loss=loss_x(j,tmp)
# 			if all_loss>tmp_loss:
# 				all_loss=tmp_loss
# 				tmp_value=j
# 		L.append(tmp_value)
# 	return L

def zhouX_global_min(dataframe):
	L=[]
	for i in range(1,2001):
		tmp=dataframe.ix[i]
		all_loss=99999999
		tmp_value=-1
		for x in range(int(min(tmp)),int(max(tmp))):
			tmp_loss=0
			for index,i in enumerate(tmp):
				tmp_loss=tmp_loss+abs(x-i)/(x+i)*w[index]
			if all_loss>tmp_loss:
				all_loss=tmp_loss
				tmp_value=x
		L.append(tmp_value)
	return L


zhou2=zhouX_global_min(dabiao_every2)
zhou3=zhouX_global_min(dabiao_every3)
zhou4=zhouX_global_min(dabiao_every4)
zhou5=zhouX_global_min(dabiao_every5)
zhou6=zhouX_global_min(dabiao_every6)
zhou7=zhouX_global_min(dabiao_every7)
zhou1=zhouX_global_min(dabiao_every1)

zhou2=map(round,zhou2)
zhou3=map(round,zhou3)
zhou4=map(round,zhou4)
zhou5=map(round,zhou5)
zhou6=map(round,zhou6)
zhou7=map(round,zhou7)
zhou1=map(round,zhou1)

zhou2=pd.Series(zhou2)
zhou3=pd.Series(zhou3)
zhou4=pd.Series(zhou4)
zhou5=pd.Series(zhou5)
zhou6=pd.Series(zhou6)
zhou7=pd.Series(zhou7)
zhou1=pd.Series(zhou1)



result=pd.concat([zhou2,zhou3,zhou4,zhou5,zhou6,zhou7,zhou1]*2,axis=1)
result=result.applymap(round)




