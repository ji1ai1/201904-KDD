import datetime
import numpy
import lightgbm
import pandas
import pickle

目錄 = ""

def predict(test_data):
	with open(目錄 + "temp/test_data", "rb") as 檔案:
		測試資料表 = pickle.load(檔案)
	with open(目錄 + "temp/test_label", "rb") as 檔案:
		測試標籤表 = pickle.load(檔案)
	with open(目錄 + "temp/test_plan", "rb") as 檔案:
		測試計劃表 = pickle.load(檔案)

	預測原表 = None
	for 子 in range(12):
		測試方式子標籤表 = 測試標籤表.merge(測試計劃表.loc[測試計劃表.計劃方式 == 子, ["會話標識"]].drop_duplicates(), on="會話標識", right_index=True)
		測試方式子標籤表["標籤"] = numpy.nan
		方式子預測表 = 測試方式子標籤表.loc[:, ["會話標識"]]
		方式子預測表["方式"] = 子
		測試方式子資料表 = 測試資料表.loc[測試方式子標籤表.index.tolist()]

		with open(目錄 + "model/" + str(子), "rb") as 檔案:
			輕模型 = pickle.load(檔案)
		方式子預測表["打分"] = 輕模型.predict(測試方式子資料表)
		del 測試方式子資料表
		預測原表 = pandas.concat([預測原表, 方式子預測表])
	預測原表.to_csv("predict", header=None, index=None, quoting=3)

	總係數 = numpy.array([0.76, 0.62, 0.59, 1.77, 2.13, 0.60, 1.31, 0.65, 1.11, 0.77, 0.94, 0.65])
	預測表 = 預測原表.copy()
	預測表["會話標識總打分"] = 預測表.groupby("會話標識")["打分"].transform("sum")
	預測表["打分"] = 預測表["打分"] / 預測表["會話標識總打分"]
	預測表 = 預測表.drop(["會話標識總打分"], axis=1)
	預測新表 = 預測表.copy()
	for 子 in range(12):
		預測新表.loc[預測新表.方式 == 子, "打分"] = 總係數[子] * 預測新表.打分[預測新表.方式 == 子]
	預測方式新表 = 預測新表.copy()
	預測方式新表["最大打分"] = 預測方式新表.groupby("會話標識")["打分"].transform("max")
	預測方式新表 = 預測方式新表.loc[預測方式新表.打分 == 預測方式新表.最大打分]
	預測方式新表 = 預測方式新表.loc[:, ["會話標識", "方式"]].drop_duplicates("會話標識").reset_index(drop=True)

	提交表 = 預測方式新表.copy()
	提交表.會話標識 = 提交表.會話標識.astype("object")
	提交表.方式 = 提交表.方式.astype("object")
	提交表.to_csv("result/result.csv", header=False, index=False, quoting=1)

	print(str(datetime.datetime.now()) + "\t結束")
