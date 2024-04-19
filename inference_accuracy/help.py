import os

bit_len = 8
test_size = 10000
step_size = 20


for i in range (int(40 / step_size)):
    cmd = ""





    cmd = ".\\a.exe " + str(bit_len) + " " + str(test_size) + " " + str(step_size) + " " + str(i * step_size)
#    print(cmd)

    os.system(cmd)






