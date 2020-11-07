import datetime
import numpy
import lightgbm
import pandas
import pickle

Prefix = ""

def predict(test_data):
	with open(Prefix + "temp/test_data", "rb") as file:
		CsDataDf = pickle.load(file)
	with open(Prefix + "temp/test_label", "rb") as file:
		CsLabelDf = pickle.load(file)
	with open(Prefix + "temp/test_plan", "rb") as file:
		CsPlansDf = pickle.load(file)

	PrediDf = None
	for A in range(12):
		CsModeALabelDf = CsLabelDf.merge(CsPlansDf.loc[CsPlansDf.JhMode == A, ["Sid"]].drop_duplicates(), on="Sid", right_index=True)
		CsModeALabelDf["Label"] = numpy.nan
		ModeAPrediDf = CsModeALabelDf.loc[:, ["Sid"]]
		ModeAPrediDf["Mode"] = A
		CsModeADataDf = CsDataDf.loc[CsModeALabelDf.index.tolist()]

		with open(Prefix + "model/" + str(A), "rb") as file:
			Mlgb = pickle.load(file)
		ModeAPrediDf["Score"] = Mlgb.predict(CsModeADataDf)
		del CsModeADataDf
		PrediDf = pandas.concat([PrediDf, ModeAPrediDf])
		PrediDf.to_csv("predict", header=None, index=None, quoting=3)

	Repe = numpy.array([0.76, 0.62, 0.59, 1.77, 2.13, 0.60, 1.31, 0.65, 1.11, 0.77, 0.94, 0.65])
	PrediDf["SidScore"] = PrediDf.groupby("Sid")["Score"].transform("sum")
	PrediDf["Score"] = PrediDf["Score"] / PrediDf["SidScore"]
	PrediDf = PrediDf.drop(["SidScore"], axis=1)
	for A in range(12):
		PrediDf.loc[PrediDf.Mode == A, "Score"] = Repe[A] * PrediDf.Score[PrediDf.Mode == A]
	PrediDf["MaxScore"] = PrediDf.groupby("Sid")["Score"].transform("max")
	PrediDf = PrediDf.loc[PrediDf.Score == PrediDf.MaxScore]
	PrediDf = PrediDf.loc[:, ["Sid", "Mode"]].drop_duplicates("Sid").reset_index(drop=True)

	SubDf = PrediDf.copy()
	SubDf.Sid = SubDf.Sid.astype("object")
	SubDf.Mode = SubDf.Mode.astype("object")
	SubDf.to_csv("result/result", header=False, index=False, quoting=1)
