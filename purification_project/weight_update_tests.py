from test_interaction_weight import *

#original weight
w = -1+1j
vconf = 1
aconf = -1
bv = 2
ba = 1.5
print("The old contribution to psi")
#print((-1)**(vconf*aconf)/np.sqrt(2))
print(np.exp(vconf*w*aconf + bv*vconf + ba*aconf))

def weight_updated(w, x, h):

    return(np.log(w)/8 - np.log(w)/2*x - np.log(w)/4*h + np.log(w)*x*h - np.log(2)/2)

w_v = weight_new("v", w)
w_a = weight_new("a", w)

res = 0
for hconf in [-1, 1]:
    #res += np.exp(weight_updated(w, vconf, hconf) + weight_updated(w, aconf, hconf))
    res += np.exp(w_v*vconf*hconf +w_a*aconf*hconf + bv*vconf + ba*aconf)

print("The new contribution to psi")
print(res)
