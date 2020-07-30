import torch
import numpy as np
print(np.version.version)

def array_equal(x, y):
    num_not_equal = 0
    for i in range(len(x.shape)):
        if x.shape[i] != y.shape[i]:
            num_not_equal+=1

    x = x.flatten()
    y = y.flatten()

    eps = np.finfo(float).eps


    for (a, b) in zip(x, y):
        if (a==np.nan or b==np.nan) and not (a==np.nan and b==np.nan):
            num_not_equal+=1
        elif np.abs(a - b) > eps:
                num_not_equal+=1

    return num_not_equal




if __name__ == "__main__":
    # mine0 = torch.load( "/home/isaac/high-res-stereo/debug/rvc/img0.pt")
    # ref0 = torch.load( "/home/isaac/high-res-stereo/debug/my_submission/img0.pt")
    # print("comparing img0.pt")
    # if array_equal(mine0, ref0) > 0:
    #     print("img0.pt does not match")
    #
    # mine1 = torch.load("/home/isaac/high-res-stereo/debug/rvc/img1.pt")
    # ref1 = torch.load("/home/isaac/high-res-stereo/debug/my_submission/img1.pt")
    # print("comparing img1.pt")
    # if array_equal(mine1, ref1) > 0:
    #     print("img1.pt does not match")
    #
    # mine2 = torch.load("/home/isaac/high-res-stereo/debug/rvc/img2.pt")
    # ref2 = torch.load("/home/isaac/high-res-stereo/debug/my_submission/img2.pt")
    # print("comparing img2.pt")
    # if array_equal(mine2, ref2) > 0:
    #     print("img2.pt does not match")
    #
    # mine3 = torch.load("/home/isaac/high-res-stereo/debug/rvc/img3.pt")
    # ref3 = torch.load("/home/isaac/high-res-stereo/debug/my_submission/img3.pt")[0]
    # print("comparing img3.pt")
    # if array_equal(mine3, ref3) > 0:
    #     print("img3.pt does not match")
    #
    # mine4 = torch.load("/home/isaac/high-res-stereo/debug/rvc/img4.pt")
    # ref4 = torch.load("/home/isaac/high-res-stereo/debug/my_submission/img4.pt")[0]
    # print("comparing img4.pt")
    # if array_equal(mine4, ref4) > 0:
    #     print("img4.pt does not match")
    #
    # mine5 = torch.load("/home/isaac/high-res-stereo/debug/rvc/img_final.pt")
    # ref5 = torch.load("/home/isaac/high-res-stereo/debug/my_submission/img_final.pt")
    # print("comparing img_final.pt")
    # if array_equal(mine5, ref5) > 0:
    #     print("img_final.pt does not match")

    mine_out = torch.load("/home/isaac/high-res-stereo/debug/rvc/out.pt").cpu()
    ref_out = torch.load("/home/isaac/high-res-stereo/debug/my_submission/out.pt").cpu()
    print("comparing out.pt")
    num_errors=array_equal(mine_out, ref_out)
    if num_errors > 0:
        print("out.pt does not match " + str(num_errors))
    print("All checks complete")