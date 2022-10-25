from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm 
import numpy as np 

dot1 =[[0.        ,0.        ,0.52022074]
,[0.35852825,0.11558893,0.04608669]
,[0.29124099,0.12733427,0.10164656]
,[0.06652473,0.00631531,0.44737081]
,[0.02645316,0.16752986,0.32623553]
,[0.2759361 ,0.01876369,0.22550259]
,[0.01729331,0.00685157,0.49608066]
,[0.30603895,0.17691468,0.03726314]
,[0.21128461,0.00265683,0.30627109]
,[0.        ,0.33714446,0.1830747 ]
,[0.30541846,0.21479023,0.        ]
,[0.02650309,0.06024921,0.43346754]
,[0.        ,0.45064731,0.06957443]
,[0.44022039,0.02449559,0.05549262]
,[0.0748851 ,0.33049245,0.11483175]
,[0.39909939,0.        ,0.12110851]
,[0.07696039,0.37320013,0.07005825]
,[0.15893297,0.25792167,0.10336734]
,[0.        ,0.51748221,0.00274054]
,[0.41177029,0.0571365 ,0.05129057]
,[0.18886876,0.33133832,0.        ]
,[0.04066523,0.07533457,0.40421195]
,[0.15721472,0.12661606,0.23638714]
,[0.24781906,0.2636097 ,0.00878531]
,[0.28285076,0.07939438,0.15796321]
,[0.50849359,0.00904545,0.00265648]
,[0.        ,0.27624027,0.24397935]
,[0.37628234,0.03350978,0.11042247]
,[0.23454869,0.22607014,0.05959181]
,[0.47173686,0.0264169 ,0.02205443]
,[0.08058797,0.06213269,0.37749323]
,[0.22645749,0.06079211,0.2329709 ]
,[0.26810609,0.21798862,0.03411487]
,[0.        ,0.50297978,0.01722521]
,[0.12018905,0.31190614,0.0881199 ]
,[0.42970864,0.        ,0.09049669]
,[0.25666698,0.08213107,0.18140391]
,[0.05895737,0.44943894,0.01182934]
,[0.02931196,0.28792468,0.20298452]
,[0.17004942,0.31606932,0.03408504]
,[0.08915152,0.43105408,0.        ]
,[0.15690345,0.        ,0.36331067]
,[0.1811358 ,0.07768749,0.26138333]
,[0.02677984,0.26676969,0.22666872]
,[0.1433787 ,0.37682562,0.        ]
,[0.06760541,0.159433  ,0.29317174]
,[0.        ,0.48066823,0.03955636]
,[0.12021144,0.09750774,0.30249932]
,[0.        ,0.49205372,0.02815343]
,[0.11886713,0.12950647,0.27184094]]
[[0.51472606,0.        ,0.        ]
,[0.00899987,0.18058981,0.32637656]
,[0.2273671 ,0.00185986,0.28542981]
,[0.09897598,0.0130204 ,0.40242702]
,[0.07694337,0.43827267,0.        ]
,[0.00525575,0.48613016,0.02362504]
,[0.07073584,0.11372586,0.33069253]
,[0.12525149,0.3891225 ,0.        ]
,[0.00702136,0.06819453,0.43922395]
,[0.19296704,0.05269569,0.26912062]
,[0.04219494,0.4731367 ,0.        ]
,[0.06232812,0.19305166,0.2592469 ]
,[0.04796338,0.02550664,0.4413619 ]
,[0.46210134,0.03899495,0.01389143]
,[0.10607954,0.19207868,0.21684393]
,[0.45402237,0.00659838,0.05297969]
,[0.13730518,0.        ,0.37720748]
,[0.06201199,0.38642258,0.06654612]
,[0.42459783,0.08979306,0.        ]
,[0.        ,0.39489807,0.12008139]
,[0.        ,0.51707018,0.        ]
,[0.        ,0.0459159 ,0.46799162]
,[0.        ,0.36747587,0.14673116]
,[0.13146577,0.02928612,0.3538788 ]
,[0.07831076,0.24255539,0.19376054]
,[0.02613809,0.34137658,0.14774032]
,[0.11242971,0.1152898 ,0.28679783]
,[0.10712268,0.33282931,0.07443305]
,[0.17615191,0.33853467,0.        ]
,[0.03238851,0.38378623,0.09915809]
,[0.10139813,0.26630927,0.14710529]
,[0.2044233 ,0.18758821,0.12244187]
,[0.01414768,0.00870009,0.4914864 ]
,[0.21790395,0.23783642,0.05927915]
,[0.13089354,0.33952873,0.04514995]
,[0.09537034,0.14755252,0.27178258]
,[0.0821488 ,0.2855586 ,0.14710529]
,[0.25774542,0.12030014,0.13675264]
,[0.36490771,0.12351273,0.02641798]
,[0.00228086,0.00220094,0.51034234]
,[0.24096757,0.10541109,0.16800389]
,[0.20997863,0.30463366,0.        ]
,[0.33206122,0.14073138,0.04187367]
,[0.39587306,0.00220832,0.11668769]
,[0.42608074,0.0428686 ,0.04595416]
,[0.19356351,0.26945667,0.0519032 ]
,[0.26468268,0.25047018,0.        ]
,[0.41942552,0.0555656 ,0.03983339]
,[0.38972988,0.03484241,0.08995714]
,[0.29768511,0.        ,0.2170806 ]]
plt.figure()
ax2 = plt.axes(projection='3d') 
ax2.set_xlim(0, 0.5)
ax2.set_ylim(0.5, 0)
ax2.set_zlim(0,0.5)




ax1 = plt.axes(projection='3d') 
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0.5, 0)
ax1.set_zlim(0,0.5)
def Z(X, Y):
    X1=(1-2*X)
    X2=1-(2*Y)/X1
    res=0.5*X1*X2
    for i in range(500):
        for j in range(500):
            if (res[i][j]<0):
                res[i][j]=0
    return res


def Z2(X, Y):
    X1=(1-2*X)
    X2=1-(2*Y)/X1
    res=0*X1*X2
    return res

x = np.arange(0, 0.5, 0.001)
y = np.arange(0, 0.5, 0.001) 
X, Y = np.meshgrid(x, y)
s1 = ax1.plot_surface(X, Y, Z(X, Y),alpha=0.1)
color1 = ['r', 'g', 'b', 'k', 'm']
marker1 = ['o', 'v', '1', 's', 'H'] 
i = 0 
for x in dot1: 
    ax1.scatter(x[0], x[1], x[2], c=color1[i], marker=marker1[i], linewidths=4) 

plt.show() 