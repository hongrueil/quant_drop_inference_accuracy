import os

bit_len = 16
test_size = 10000
step_size = 1


# for i in range (int(40 / step_size)):
#     cmd = ""





#     cmd = ".\\a.exe " + str(bit_len) + " " + str(test_size) + " " + str(step_size) + " " + str(i * step_size) + " " + str(0)
#     #print(cmd)

#     os.system(cmd)



ll = [0, 1, 2, 5, 10, 20]

for i in range(6):
    cmd = ""
    cmd = ".\\a.exe " + str(bit_len) + " " + str(test_size) + " " + str(step_size) + " " + str(0) + " "+ str(i) + " > ./quant_output_new/16_bit/accuracy_order/" +"accuracy_order_" +str(ll[i]) +"th_output.txt"
    print(cmd)
    os.system(cmd)

