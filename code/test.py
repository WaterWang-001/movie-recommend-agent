import sys
import os
import re
import json
from datetime import datetime
from llm_demo_updated import main
from typing import List
import pandas as pd
from thefuzz import process, fuzz

# 定义 Tee 类，用于重定向输出
class Tee:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout  # 保存终端的标准输出
        self.log = open(log_file_path, "a", encoding="utf-8")  # 打开日志文件

    def write(self, message):
        self.terminal.write(message)  # 输出到终端
        self.log.write(message)  # 输出到日志文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 初始化日志文件
log_file_path = os.path.join(os.getcwd(), f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
sys.stdout = Tee(log_file_path)  # 重定向标准输出

# Helper function to calculate recall and hit rate
def calculate_metrics_for_pairs(pairs):
    def calculate_recall(predicted_list, true_list):
        predicted_set = set(predicted_list)
        true_set = set(true_list)
        relevant_retrieved = predicted_set.intersection(true_set)
        recall = len(relevant_retrieved) / len(true_set) if len(true_set) > 0 else 0
        hit = 1 if len(relevant_retrieved) > 0 else 0
        return recall, hit

    recall_scores = []
    hit_scores = []
    for pred, true in pairs:
        recall, hit = calculate_recall(pred, true)
        recall_scores.append(recall)
        hit_scores.append(hit)

    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    hit_rate = sum(hit_scores) / len(hit_scores) if hit_scores else 0

    return recall_scores, average_recall, hit_rate

# Function to convert movie titles to IDs
def movie_to_ids(movie_list: List[str]) -> List[int]:
    movie_data = pd.read_csv('/Users/sizhewang/课程/UCB CS 194/ml-1m/movies.dat', sep='::', engine='python', encoding='latin-1', names=["MovieID", "Title", "Genres"])
    movie_list_df = pd.DataFrame(movie_list, columns=["Title"])
    movie_list_df["Title"] = movie_list_df["Title"].apply(
        lambda x: process.extractOne(x, movie_data["Title"], scorer=fuzz.ratio)[0]
    )
    data = pd.merge(movie_data, movie_list_df, on="Title", how="right")
    movie_ids = data["MovieID"].tolist()
    return movie_ids



test = {
    "userid": [297, 409, 711, 743, 866, 1143, 1273, 2054,2095,2154,2317,3128,3695,3950,4924,5286,5530,5594,5759],
    "previous_movie_seq": [
        [818, 111, 1012, 3703, 1020, 3897, 1288, 2791, 1136, 1294],    # User 1's previous movie sequence
        [2076, 36, 1097, 195, 1982, 2023, 1196, 1198, 858, 260],  # User 2's previous movie sequence
        [1994, 431, 1721, 1193, 2764, 3925, 922, 930, 1617, 541],         # User 3's previous movie sequence
        [541, 455, 1193, 3793, 551, 3897, 3911, 3952, 3916, 3882],
        [1210, 1474, 2628, 2028, 1196, 3745, 3793, 3624, 3827, 3623],
        [2739, 589, 1608, 231, 1210, 1084, 1646, 1610, 1949, 1006],
        [1614, 7, 3276, 1198, 1527, 1573, 141, 2987, 1252, 1260],
        [1682, 257, 1968, 3868, 544, 3764, 1270, 1288, 2791, 2795],
        [1603, 1721, 2716, 3072, 3450, 1097, 1284, 266, 110, 1035],
        [337, 1210, 2985, 1250, 3101, 364, 3250, 750, 2858, 1148],
        [1193, 3717, 2797, 527, 852, 3399, 2023, 3932, 3916, 2858],
        [1193, 2145, 1378, 1036, 593, 3859, 3897, 3826, 3752, 2858],
        [858, 3098, 260, 2072, 1619, 2138, 2174, 2797, 1073, 2968],
        [2089, 527, 1210, 810, 2706, 3082, 3298, 3452, 2858, 3186],
        [778, 527, 598, 1196, 1594, 82, 1097, 3649, 912, 923],
        [908, 3447, 1552, 2558, 1085, 3022, 3462, 3629, 2788, 1256],
        [260, 2147, 1968, 2005, 236, 292, 3111, 1230, 497, 2792],
        [587, 1246, 2380, 1193, 3207, 2683, 2336, 913, 2805, 1959],
        [260, 988, 527, 2628, 1945, 858, 153, 3052, 2883, 2959]
    ],
    "future_movie_seq": [
        [2858, 2324, 2396, 1734, 1784, 2599, 1235, 3361, 858, 1221, 1193, 923, 593, 919, 527, 1247, 908, 296, 608, 1213], 
        [1214, 2028, 589, 1387, 2571, 457, 1233, 1610, 110, 1036, 1200, 1222, 1240, 1291, 2194, 1912, 3107, 1954, 2000, 6], 
        [1179, 1344, 903, 904, 1086, 924, 293, 1953, 1674, 2762, 2212, 1343, 457, 2871, 2183, 2176, 1127, 1129, 3176, 2916], 
        [3481, 3751, 3786, 1252, 1230, 969, 3307, 2248, 1172, 2396, 1265, 1307, 1197, 3536, 2065, 1393, 342, 2502, 1721, 3567], 
        [3821, 3189, 3286, 3785, 1721, 2997, 2858, 3147, 3114, 588, 1, 1566, 1907, 364, 2687, 2092, 783, 239, 2142, 48], 
        [1265, 356, 2628, 260, 858, 1235, 3733, 1234, 1387, 1271, 1198, 1957, 1302, 1097, 1270, 3361, 2020, 2243, 2150, 1231], 
        [1267, 1617, 3683, 541, 2066, 1284, 930, 1179, 2186, 978, 3629, 2351, 858, 3730, 2935, 1212, 923, 947, 2019, 922], 
        [1079, 2804, 1259, 1197, 3361, 2918, 1220, 1285, 1663, 1307, 2000, 1238, 1257, 3039, 3608, 3525, 2797, 2423, 3526, 2001], 
        [2948, 3510, 2791, 3623, 3916, 3948, 3624, 3578, 3753, 3189, 3408, 3831, 3555, 3755, 3326, 3827, 3798, 3618, 3301, 3744], 
        [1193, 527, 720, 2997, 2908, 1225, 1208, 1247, 1617, 1945, 1267, 593, 908, 3629, 1219, 1262, 1222, 1223, 1080, 1235], 
        [2572, 3534, 2997, 2683, 3298, 3516, 2770, 3481, 3555, 3753, 3513, 1284, 608, 110, 223, 2355, 2771, 2976, 2369, 2541], 
        [2396, 2997, 1537, 2700, 728, 2542, 1641, 2355, 1923, 3060, 2395, 1060, 3174, 3253, 2324, 344, 785, 1580, 543, 588], 
        [1097, 2628, 2100, 2140, 2161, 2173, 2193, 367, 2005, 3479, 2143, 1525, 3489, 1126, 924, 1196, 1200, 1214, 589, 1270], 
        [2976, 3081, 2889, 3176, 3180, 3301, 3276, 3324, 3355, 3146, 3114, 1265, 2108, 2542, 3052, 1885, 2321, 180, 235, 1060], 
        [3224, 3342, 3083, 318, 1230, 1213, 1267, 3160, 1094, 1225, 593, 1617, 2396, 608, 3543, 50, 2028, 1247, 1299, 1041], 
        [2239, 2335, 3097, 1230, 2300, 3037, 3543, 1244, 933, 1276, 2289, 3363, 905, 921, 947, 2997, 1077, 3424, 2863, 3114], 
        [3672, 3673, 3705, 3686, 3713, 3620, 3623, 3691, 3668, 3624, 3699, 3712, 858, 1242, 1387, 2028, 1954, 2692, 457, 2912], 
        [3219, 2013, 2692, 2396, 2881, 2716, 2776, 154, 3094, 858, 923, 608, 306, 260, 1225, 593, 1198, 919, 1233, 527], 
        [3125, 3175, 2912, 2712, 2829, 1196, 1198, 1218, 1221, 3265, 1387, 1197, 1240, 1277, 2571, 1200, 2947, 2028, 589, 1304]
    ]
}

contents = []
for i in range(len(test["userid"])):
    print(f"Processing user {test['userid'][i]} ({i + 1}/{len(test['userid'])})")
    prompt = f"User{test['userid'][i]} has watched movie {test['previous_movie_seq'][i]}, Can you recommend some movie for him/her?"
    contents.append(prompt)

results = []
for i, content in enumerate(contents):
    print(f"Generating recommendations for user {test['userid'][i]}...")
    predicted = movie_to_ids(main(content))
    results.append([predicted, test["future_movie_seq"][i]])
    print(f"Predicted: {predicted}")
    print(f"Future sequence: {test['future_movie_seq'][i]}")

# Calculate metrics
recall_scores, average_recall, hit_rate = calculate_metrics_for_pairs(results)
print("\nResults:")
print("Individual Recall Scores:", recall_scores)
print("Average Recall:", average_recall)
print("Hit Rate:", hit_rate)

print(f"\nAll output saved to log file: {log_file_path}")


