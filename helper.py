# combine segmented segments to complete ones
# generate new_training_data_.txt

# file = open("./data/" + name + "_data_0810.txt")
# line = file.readline().strip()
# new_f = open("./data/new_training_data_.txt", "w+")
#
# while line != "":
#     tmp = line.split("\t")
#     line = file.readline().strip()
#     unormalized = "".join(tmp[1].split(" "))
#     normalized = "".join(tmp[2].split(" "))
#     new_f.write(unormalized + "\t" + normalized + "\n")
#
# new_f.close()
# file.close()
#
# file = open("./data/new_training_data_.txt", "r")
# data1 = open("./data/training_data.txt", "w+")
# data2 = open("./data/testing_data.txt", "w+")
#
# line = file.readline()
# cnt = 0
# while line != "":
#     if cnt < 15000:
#         data1.write(line)
#     elif cnt < 20000:
#         data2.write(line)
#     else:
#         break
#     line = file.readline()
#     cnt += 1
# file.close()
# data1.close()
# data2.close()

key_set = ["train", "test"]
# key_set = ["training", "validation", "test"]

for key in key_set:
    prex = "_data_0823"
    # prex = "_dynamic_data"
    file = open("./data/exp0823/data_augment_" + key + ".txt")
    line = file.readline().strip()
    new_f = open("./data/" + key + prex + "_des.txt", "w+")
    new_f_o = open("./data/" + key + prex + "_opr.txt", "w+")

    while line != "":
        tmp = line.split("\t")
        line = file.readline().strip()
        label = tmp[0]
        unormalized_d = tmp[1]
        normalized_d = tmp[2]
        new_f.write(label + "\t" + unormalized_d + "\t" + normalized_d + "\n")

        label2 = tmp[3]
        unormalized_o = tmp[4]
        normalized_o = tmp[5]
        new_f_o.write(label2 + "\t" + unormalized_o + "\t" + normalized_o + "\n")

    new_f.close()
    new_f_o.close()
    file.close()

# # generate dynamic dataset
# import random

# name = "validation"
# file_name = "./data/exp0803/" + name + "_data_0803.lpy.csv"
# file_t = open(file_name)
# line = file_t.readline()
#
# res_file = open("./data/exp0803/" + name + "_dynamic_data.txt", "w+")
# cnt = 0
# while line != "":
#     cnt += 1
#     if cnt > 80000:
#         break
#     res = line.split("\t")
#     random_n = random.random()
#     if random_n < 0.6:
#         res_file.write(line)
#     else:
#         res_file.write("\t".join(res[:3]) + "\n")
#     line = file_t.readline()
# res_file.close()

########### Analyze results of models

# file2 = open("./runs/Exp/Single_task11502361344/right_cases.txt")
# file1 = open("./runs/Exp/Single_task11502361227/right_cases.txt")
#
# line = file1.readline()
# arr1 = []
# while line != "":
#     arr1.append(line)
#     line = file1.readline()
# line = file2.readline()
#
# arr2 = []
# while line != "":
#     if line in arr1:
#         arr1.remove(line)
#         arr2.append(line)
#     line = file2.readline()
#
# ans = open("ans.txt", "w+")
# ans2 = open("ans_overlap.txt", "w+")
# for i in arr1:
#     ans.write(i)
#
# for i in arr2:
#     ans2.write(i)
# ans.close()
# ans2.close()
