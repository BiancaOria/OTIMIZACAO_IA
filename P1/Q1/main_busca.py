from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt



# grs = GlobalRandomSearch(1000,.5)
# grs.search()

# lrs = LocalRandomSearch(1000,0.5)
# lrs.search()
# plt.show()
hill = Hill(1000,0.5,20)
hill.search()