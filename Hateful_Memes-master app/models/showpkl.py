# #show_pkl.py
 
# import pickle
# path='logistic.pkl' #path='/root/……/aus_openface.pkl' Path where the pkl file is located
	   
# f=open(path,'rb')
# data=pickle.load(f)
 
# print(data)
# #print(len(data))
import sys
sys.getdefaultencoding()
import pickle
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = './tfid_labels.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
#fr.close()
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'w')
ft.write(inf)
ft.write("\n")
np.set_printoptions(threshold=1000000000000000)
path = './tfid_objects.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'a')
ft.write(inf)
ft.write("\n")
path = './tfid_text.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'a')
ft.write(inf)
ft.write("\n")
path = './ada.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'a')
ft.write(inf)
ft.write("\n")
path = './logistic.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'a')
ft.write(inf)
ft.write("\n")
path = './nai.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'a')
ft.write(inf)
ft.write("\n")
path = './rfc.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'a')
ft.write(inf)
ft.write("\n")
path = './top_logistic.pkl'
file = open(path,'rb')
inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
print(inf)
inf=str(inf)
obj_path = './textfile.txt'
ft = open(obj_path, 'a')
ft.write(inf)
ft.write("\n")