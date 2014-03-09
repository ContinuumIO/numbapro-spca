import numpy as np

p = 1000
n = 100
X = np.random.randn(n,p)

A = (1./n)*X.T.dot(X)
U,S,tmp = np.linalg.svd(A)
A = U.dot(np.diag(S).dot(np.diag(1./np.arange(1,p+1)))).dot(np.conjugate(U.T))

del p,n,X,U,S
#---------------------------------------------

epsilon = 0.1
d = 3
k = 10
p = A.shape[0]

U,S,tmp = np.linalg.svd(A)

#Vd = U[:,1:d+1].dot(np.diag(np.sqrt(S[1:d+1])))
Vd = U[:,0:d].dot(np.diag(np.sqrt(S[0:d])))
numSamples = (4./epsilon)**d

##actual algorithm
opt_x = np.zeros((p,1))
opt_v = -np.inf

#GENERATE ALL RANDOM SAMPLES BEFORE
C = np.random.randn(d,numSamples)

for i in np.arange(1,numSamples+1):
	
	#c = np.random.randn(d,1)
	#c = C[:,i-1]
	c = C[:,i-1:i]
	c = c/np.linalg.norm(c)
	a = Vd.dot(c)

	#partial argsort in numpy?
	#if partial, kth largest is p-k th smallest
	#but need indices more than partial
	I = np.argsort(a, axis=0)
	val = np.linalg.norm(a[I[-k:]]) #index backwards to get k largest

	if val > opt_v:
		opt_v = val
		opt_x = np.zeros((p,1))
		#print((opt_x[I[0:k]]).shape)
		#print((a[I[0:k]]/val).shape)
		opt_x[I[-k:]] = a[I[-k:],:]/val


print(np.linalg.norm(np.conjugate(Vd.T).dot(opt_x)))
print(np.conjugate(opt_x.T).dot(A.dot(opt_x)))
