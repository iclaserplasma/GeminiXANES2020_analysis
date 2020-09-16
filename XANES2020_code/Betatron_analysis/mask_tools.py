from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
import numpy as np
def reorder_labels(labelled):
    reg_ps = regionprops(labelled)
    Ny,Nx = np.shape(labelled)
    c_x = np.array([x.centroid[0] for x in reg_ps])
    c_y = np.array([x.centroid[1] for x in reg_ps])
    c_l = np.array([x.label for x in reg_ps])
    reg00_ind = np.argmin(c_x**2+c_y**2)
    reg01_ind = np.argmin((Nx-c_x)**2+c_y**2)
    reg10_ind = np.argmin(c_x**2+(Ny-c_y)**2)
    reg11_ind = np.argmin((Nx-c_x)**2+(Ny-c_y)**2)
    dx = c_x[reg01_ind]-c_x[reg00_ind]
    dy = c_y[reg01_ind]-c_y[reg00_ind]
    dr = np.sqrt(dx**2+dy**2)
    X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    c_u = c_x*dx/dr + c_y*dy/dr
    c_v = -c_x*dy/dr + c_y*dx/dr
    C_fit = []
    K_list = np.arange(4,20)
    for N_c in K_list:

        KM = KMeans(n_clusters=N_c).fit(c_u.reshape(-1,1))
        C_fit.append(KM.inertia_/np.min(np.diff(np.sort(KM.cluster_centers_.flatten()))))

    N_cu = K_list[np.argmin(C_fit*K_list)]
    KM_u = KMeans(n_clusters=N_cu).fit(c_u.reshape(-1,1))

    C_fit = []
    for N_c in K_list:

        KM = KMeans(n_clusters=N_c).fit(c_v.reshape(-1,1))
        C_fit.append(KM.inertia_/np.min(np.diff(np.sort(KM.cluster_centers_.flatten()))))

    N_cv = K_list[np.argmin(C_fit*K_list)]
    KM_v = KMeans(n_clusters=N_cv).fit(c_v.reshape(-1,1))

    x = np.sort(KM_u.cluster_centers_.flatten())
    y = np.sort(KM_v.cluster_centers_.flatten())
    X,Y = np.meshgrid(x,y)
    X = X.flatten(order='F')
    Y = Y.flatten(order='F')
    # plt.plot(c_u,c_y,'.')
    # plt.plot(KM.cluster_centers_,np.ones(N_c)*1000,'+')

    # plt.plot(KM_u.cluster_centers_.flatten(),KM_v.cluster_centers_.flatten(),'+')
    iOrder = []
    labelled_reordered = labelled*0
    for n,(x,y) in enumerate(zip(X,Y)):
        iSel = np.argmin((c_u-x)**2+(c_v-y)**2)
        iOrder.append(c_l[iSel])
        labelled_reordered[labelled==iOrder[-1]] = n+1
    return labelled_reordered
