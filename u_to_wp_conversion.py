import numpy as np
import scipy.optimize as sp
import optimization

def get_wp(results):
    u1 = results[-1].u1
    u2 = results[-1].u2
    u3 = results[-1].u3
    u4 = results[-1].u4

    a1 = solving([0,0,0], u1)
    a2 = solving([0,0,0], u2)
    a3 = solving([0,0,0], u3)
    a4 = solving([0,0,0], u4)

    print("\n")
    print("Angles for unitary compensation:")
    print(f"arya.set_sample_angles({a1})")
    print(f"bran.set_sample_angles({a2})")
    print(f"cersei.set_sample_angles({a3})")
    print(f"dany.set_sample_angles({a4})")
    print("\n")

    
def get_u(results):
    print("\n")
    print("The unitary compensation:")
    print(f"u_arya = np.{repr(results[-1].u1)}")
    print(f"u_bran = np.{repr(results[-1].u2)}")
    print(f"u_cersei = np.{repr(results[-1].u3)}")
    print(f"u_dany = np.{repr(results[-1].u4)}")


def unitary(angle, u):

    a = angle[0]
    b = angle[1]
    y = angle[2]
    
    f = (1/2)*(-np.cos(2*(a-b))-np.cos(2*(b-y))) - np.real(u[0][0])
    g = (1/2)*(np.sin(2*(a - b)) + np.sin(2*(b - y))) - np.real(u[0][1])
    h = (1/2)*(-np.sin(2*(a - b)) - np.sin(2*(b - y))) - np.real(u[1][0])
    v = (1/2)*(-np.cos(2*(a-b))- np.cos(2*(b - y))) - np.real(u[1][1])

    K = (1/2)*(-np.cos(2*b) + np.cos(2*(a - b + y))) - np.imag(u[0][0])
    m = (1/2)*(-np.sin(2*b) + np.sin(2*(a - b + y))) - np.imag(u[0][1])
    z = (1/2)*(-np.sin(2*b) + np.sin(2*(a - b + y)))- np.imag(u[1][0])
    e = (1/2)*(np.cos(2*b) - np.cos(2*(a - b + y))) - np.imag(u[1][1])

    return  (f,g,h,v,K,m,z,e)


def solving(angle, u):
    result = sp.least_squares(unitary, angle,method='trf', args=[u], max_nfev=1000000000)
    QWP1 = result.x[0]
    HWP1 = result.x[1]
    QWP2 = result.x[2]
    return([QWP2*180/np.pi,HWP1*180/np.pi,QWP1*180/np.pi])