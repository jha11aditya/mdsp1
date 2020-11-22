import sys
import os
import numpy as np
from numpy.linalg import eigh
from numpy.linalg import eig
import glob

imgs = list()
for f in glob.glob("input/*.ppm"):
    imgs.append(f)



form = ""
n,m = 0,0
ran = ""

A = list()

### Loading all 30 images' data into list A
for img in imgs:
    fil = open(img,'r')
    lin = fil.readlines()


    form = lin[0]
    n,m = map(int,lin[1].split(" "))
    ran = lin[2]

    li = list()
    for i in range(3,len(lin)):
        li.append(list(map(int,lin[i].strip("\n").rstrip().split(" "))))

    flatli = list(np.concatenate(li).flat)

    dat = [ 0 for i in range(m*n)]

    ind=0
    for i in range(0,len(flatli)-2,3):
        dat[ind] = flatli[i]
        ind+=1
    A.append(dat)

    fil.close()


### Average of 30 data points
print("Anxp:",len(A),len(A[0]))
avgli = np.mean(A,axis=1)
# stdevli = np.std(A, axis=1)



### Centering each image data
for i in range( len(A) ):
    for j in range(len(A[0])):
        A[i][j] = (A[i][j]-avgli[i])

print("check center",sum([i for i in A[1]]))



### Covariance of data variables
covA = np.cov(A,rowvar=False)
print("CovarA shape", covA.shape)

### Eigen Vaue Decomposition for symmetric matrix
e_valli,e_vecli = eigh(covA)



### Printing Proportion of Variances first 16
esum = sum(e_valli)
pr = [ ev/esum for ev in e_valli   ]
keys = np.flip(np.argsort( np.array(e_valli) ))
print("Proportion of Variances Top 16")
for k in keys[:16]:
    print("eval: ", e_valli[k], " proportion: ", pr[k])
### Reducing to top n_comp components 
ncomp = int(sys.argv[1])
inds = np.flip( np.argsort(e_valli) )
inds = inds[:ncomp]
e_valli = e_valli[inds]
e_vecli = e_vecli[:,inds]

### Total Proportion of Variances for n_comp components
sc = 0
for ev in e_valli:
    sc += ev/esum
print("Total Proportion of Variances Top ",ncomp," components: ", sc)




A = np.array(A)
print("A shape", A.shape)
print("e_vec shape", e_vecli.shape)


### Projection of data matrix A onto extracted feauture space E', Z = A . E'
Z = np.matmul(A,e_vecli)
print("Z_shape",Z.shape )
### Reconstruction of data from score Z, A' = Z . E' = A . E . E'
new_A = np.matmul(Z, e_vecli.T)
print("new_A  shape", new_A.shape)

new_A = list(new_A)


### Re Adding means for all 30 data points
for i in range( len(new_A) ):
    for j in range(len(new_A[0])):
        new_A[i][j] += avgli[i]




### Writing new reconstructed images
import os
if not os.path.exists('recon_out'):
    os.makedirs('recon_out')

for i in range(len(imgs)):
    ofil = open("recon_out/recon_"+str(imgs[i].split("/")[1]), "w")
    print("Converted: ",i ,imgs[i].split("/")[1])
    ofil.write(form)
    ofil.write(str(n) + " " + str(m) + "\n")
    ofil.write(ran)
    for ele in list(new_A[i]):
        ofil.write(str(int(ele))+" "+str(int(ele))+" "+str(int(ele))+"\n")
    
    ofil.close()




