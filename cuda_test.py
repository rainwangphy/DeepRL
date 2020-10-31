import torch
import numpy as np

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Yes")
        np_vec = np.array([2, 3, 4])
        ten = torch.from_numpy(np_vec)
        print(ten.cuda())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name())
        print("test")
