import numpy as np



def main():
    data = np.load('info.npz')
    print(data['e_data'].shape)
main()