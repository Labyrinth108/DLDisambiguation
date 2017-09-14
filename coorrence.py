# !/usr/bin/env python
# coding=utf-8

# # discover relationships between disease and procedure from labeled_data and save in coorrence_file

file_des_opr = open("./data/exp0823/filter_pos_data_review_final.txt")
res = open("coorrence_file.txt", "w+")

entity_names = []
map_dict = []

line = file_des_opr.readline()
while line != "":
    tmp = line.split("\t")
    des_e_name = tmp[1].strip()
    opr_e_name = tmp[3].strip()

    if entity_names.__contains__(des_e_name):
        i = entity_names.index(des_e_name)
        if opr_e_name in map_dict[i].keys():
            map_dict[i][opr_e_name] += 1
        else:
            map_dict[i][opr_e_name] = 1
    else:
        entity_names.append(des_e_name)
        map_dict.append({opr_e_name: 1})
    line = file_des_opr.readline()

length = len(entity_names)
for i in range(length):
    res.write(entity_names[i] + "\t")
    map_dict_des = map_dict[i]
    for k, v in map_dict_des.items():
        res.write(k + ":" + str(v) + "_")
    res.write("\n")

# # discover relationships between disease and procedure from Database and save in new_co_file.file

# # !/usr/bin/env python
# # coding=utf-8
# import MySQLdb
# import codecs

# conn = MySQLdb.connect("localhost", "root", "10081008", "medical", charset='utf8')
# cursor = conn.cursor()
# cursor.execute('select S050100, S050501 from d2014_2015 where S050100 != "" and S050501 != "" limit 10000000;')
# values = cursor.fetchall()
# print("Finished data loading...")
#
# cursor.execute('select 疾病名称 from Norm6;')
# disease_tuple = cursor.fetchall()
# disease_list = [i[0] for i in disease_tuple]
#
# cursor.execute('select 手术名称 from Treatment;')
# operation_tuple = cursor.fetchall()
# operation_list = [i[0] for i in operation_tuple]
# print("Finished Disease and Operation Names loading...")
#
# co_file = codecs.open("./data/new_co_file.txt", "w+", encoding="utf-8")
# map_dict = {}
# for i in values:
#     d_name = i[0]
#     o_name = i[1]
#     if d_name in disease_list and o_name in operation_list:
#         if d_name in map_dict.keys():
#             o_dict = map_dict[d_name]
#             if o_name in o_dict.keys():
#                 map_dict[d_name][o_name] += 1
#             else:
#                 map_dict[d_name][o_name] = 1
#         else:
#             map_dict[d_name] = {o_name: 1}
#
# for k, v in map_dict.iteritems():
#     co_file.write(k + "\t")
#     for o_name, num in v.iteritems():
#         co_file.write(o_name + ":"+str(num) + "_")
#     co_file.write("\n")
# co_file.close()
