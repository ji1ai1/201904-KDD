import datetime
import gc
import numpy
import json
import lightgbm
import math
import pandas
import pickle
import random
import resource

S, H = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (35 * 1024 * 1024 * 1024, H))

Prefix = ""
def train():
	def GetChsh(a, b):
		if b >= 38:
			return 1
		if b >= 29 and b < 33:
			return 2
		if b >= 22.9 and b < 25:
			return 3
		if b < 22.9 and a < 113.72:
			return 3
		if b < 22.9 and a >= 113.72:
			return 4

		return 0


	def CalUero(a, b, c, d):
		aa = math.sin(math.pi * (b - d) / 360) ** 2
		bb = math.sin(math.pi * (a - c) / 360) ** 2
		cc = math.cos(math.pi * b / 180)
		dd = math.cos(math.pi * d / 180)

		return 2 * 6378137 * math.asin((aa + bb * cc * dd) ** 0.5)


	PDf = pandas.read_csv(Prefix + "data/profiles.csv", dtype=numpy.int32)
	PDf = PDf.rename({"pid": "P"}, axis=1)

	CsQueriesDf = pandas.read_csv(Prefix + "data/test_queries.csv", header=0, names=["Sid", "P", "QiTime", "Qd", "Zd"], dtype={"Sid": numpy.int32, "P": numpy.float32, "QiTime": object, "Qd": object, "Zd": object})
	CsQueriesDf = CsQueriesDf.fillna(-1)
	CsQueriesDf["P"] = CsQueriesDf.P.astype(numpy.int32)
	CsQueriesDf["Qisi"] = pandas.Series([datetime.datetime.strptime(A, "%Y-%m-%d %H:%M:%S").timestamp() for A in CsQueriesDf.QiTime]).astype(numpy.int32)
	CsQueriesDf["QiYoubi"] = (823543 + (CsQueriesDf.Qisi - 57600 - 14400) // 86400) % 7
	CsQueriesDf["Qidi"] = (CsQueriesDf.Qisi - 57600) // 86400 - 17865
	CsQueriesDf["Qihi"] = (CsQueriesDf.Qisi - 57600) // 3600 - 428760
	CsQueriesDf["QdLong"] = pandas.Series([a.split(",")[0] for a in CsQueriesDf.Qd], dtype=numpy.float32)
	CsQueriesDf["QdLati"] = pandas.Series([a.split(",")[1] for a in CsQueriesDf.Qd], dtype=numpy.float32)
	CsQueriesDf["ZdLong"] = pandas.Series([a.split(",")[0] for a in CsQueriesDf.Zd], dtype=numpy.float32)
	CsQueriesDf["ZdLati"] = pandas.Series([a.split(",")[1] for a in CsQueriesDf.Zd], dtype=numpy.float32)
	CsQueriesDf["QzDista"] = pandas.Series([CalUero(QdLong, QdLati, ZdLong, ZdLati) for QdLong, QdLati, ZdLong, ZdLati in zip(CsQueriesDf.QdLong, CsQueriesDf.QdLati, CsQueriesDf.ZdLong, CsQueriesDf.ZdLati)]).astype(numpy.float32)
	CsQueriesDf["Chsh"] = pandas.Series([GetChsh(Long, Lati) for Long, Lati in zip(CsQueriesDf.QdLong, CsQueriesDf.QdLati)]).astype(numpy.int16)
	CsQueriesDf["ChshP"] = [str(Chsh) + "_" + str(P) for Chsh, P in zip(CsQueriesDf.Chsh, CsQueriesDf.P)]
	CsQueriesDf = CsQueriesDf.drop(["QiTime"], axis=1)

	CsPlansDf = pandas.read_csv(Prefix + "data/test_plans.csv", header=0, names=["Sid", "JiTime", "PlansList"], dtype={"Sid": numpy.int32, "JiTime": object, "PlansList": object})
	CsPlansDf["Jisi"] = pandas.Series([datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S").timestamp() for a in CsPlansDf.JiTime], dtype=numpy.int32)
	CsPlansDf = CsPlansDf.drop(["JiTime"], axis=1)

	CsDf = CsQueriesDf
	CsDf = CsDf.merge(CsPlansDf, on="Sid", how="left")
	del CsQueriesDf
	del CsPlansDf

	XlQueriesDf = pandas.concat([
		pandas.read_csv(Prefix + "data/train_queries_phase1.csv", header=0, names=["Sid", "P", "QiTime", "Qd", "Zd"], dtype={"Sid": numpy.int32, "P": numpy.float32, "QiTime": object, "Qd": object, "Zd": object})
		, pandas.read_csv(Prefix + "data/train_queries_phase2.csv", header=0, names=["Sid", "P", "QiTime", "Qd", "Zd"], dtype={"Sid": numpy.int32, "P": numpy.float32, "QiTime": object, "Qd": object, "Zd": object})
	]).reset_index(drop=True)
	XlQueriesDf = XlQueriesDf.fillna(-1)
	XlQueriesDf["P"] = XlQueriesDf.P.astype(numpy.int32)
	XlQueriesDf["Qisi"] = pandas.Series([datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S").timestamp() for a in XlQueriesDf.QiTime]).astype(numpy.int32)
	XlQueriesDf["QiYoubi"] = (823543 + (XlQueriesDf.Qisi - 57600 - 14400) // 86400) % 7
	XlQueriesDf["Qidi"] = (XlQueriesDf.Qisi - 57600) // 86400 - 17865
	XlQueriesDf["Qihi"] = (XlQueriesDf.Qisi - 57600) // 3600 - 428760
	XlQueriesDf["QdLong"] = pandas.Series([a.split(",")[0] for a in XlQueriesDf.Qd], dtype=numpy.float32)
	XlQueriesDf["QdLati"] = pandas.Series([a.split(",")[1] for a in XlQueriesDf.Qd], dtype=numpy.float32)
	XlQueriesDf["ZdLong"] = pandas.Series([a.split(",")[0] for a in XlQueriesDf.Zd], dtype=numpy.float32)
	XlQueriesDf["ZdLati"] = pandas.Series([a.split(",")[1] for a in XlQueriesDf.Zd], dtype=numpy.float32)
	XlQueriesDf["QzDista"] = pandas.Series([CalUero(QdLong, QdLati, ZdLong, ZdLati) for QdLong, QdLati, ZdLong, ZdLati in zip(XlQueriesDf.QdLong, XlQueriesDf.QdLati, XlQueriesDf.ZdLong, XlQueriesDf.ZdLati)], dtype=numpy.float32)
	XlQueriesDf["Chsh"] = pandas.Series([GetChsh(Long, Lati) for Long, Lati in zip(XlQueriesDf.QdLong, XlQueriesDf.QdLati)], dtype=numpy.int16)
	XlQueriesDf["ChshP"] = [str(Chsh) + "_" + str(P) for Chsh, P in zip(XlQueriesDf.Chsh, XlQueriesDf.P)]
	XlQueriesDf = XlQueriesDf.drop(["QiTime"], axis=1)

	XlClicksDf = pandas.concat([
		pandas.read_csv(Prefix + "data/train_clicks_phase1.csv", header=0, names=["Sid", "DjTime", "DjMode"], dtype={"Sid": numpy.int32, "JiTime": object, "PlansList": object})
		, pandas.read_csv(Prefix + "data/train_clicks_phase2.csv", header=0, names=["Sid", "DjTime", "DjMode"], dtype={"Sid": numpy.int32, "JiTime": object, "PlansList": object})
	]).reset_index(drop=True)
	XlClicksDf["DjSi"] = pandas.Series([datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S").timestamp() for a in XlClicksDf.DjTime], dtype=numpy.int32)
	XlClicksDf = XlClicksDf.drop(["DjTime"], axis=1)

	XlPlansDf = pandas.concat([
		pandas.read_csv(Prefix + "data/train_plans_phase1.csv", header=0, names=["Sid", "JiTime", "PlansList"], dtype={"Sid": numpy.int32, "JiTime": object, "PlansList": object})
		, pandas.read_csv(Prefix + "data/train_plans_phase2.csv", header=0, names=["Sid", "JiTime", "PlansList"], dtype={"Sid": numpy.int32, "JiTime": object, "PlansList": object})
	]).reset_index(drop=True)
	XlPlansDf["Jisi"] = pandas.Series([datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S").timestamp() for a in XlPlansDf.JiTime], dtype=numpy.int32)
	XlPlansDf = XlPlansDf.drop(["JiTime"], axis=1)

	XlDf = XlQueriesDf
	XlDf = XlDf.merge(XlClicksDf, on="Sid", how="left")
	XlDf = XlDf.merge(XlPlansDf, on="Sid", how="left")
	XlDf.DjMode = XlDf.DjMode.fillna(0)
	del XlQueriesDf
	del XlClicksDf
	del XlPlansDf
	gc.collect()

	TreDf = pandas.concat([
		CsDf.loc[:, ["Sid", "P", "ChshP", "Qd", "Zd", "QdLong", "QdLati", "ZdLong", "ZdLati", "Chsh", "Qisi", "QiYoubi", "Qidi", "Qihi", "PlansList"]]
		, XlDf.loc[:, ["Sid", "P", "ChshP", "Qd", "Zd", "QdLong", "QdLati", "ZdLong", "ZdLati", "Chsh", "Qisi", "QiYoubi", "Qidi", "Qihi", "PlansList"]]
	]).reset_index(drop=True)

	TrePlansDf = TreDf[~TreDf.PlansList.isna()].reset_index(drop=True)
	TrePlap = [json.loads(a) for a in TrePlansDf.PlansList]
	TrePlas = [len(a) for a in TrePlap]
	TrePlap = [b for a in TrePlap for b in a]
	TrePlansDf = TrePlansDf.loc[:, ["Sid", "P", "ChshP", "Qd", "Zd", "QiYoubi", "Qidi", "Qihi"]].merge(pandas.DataFrame({
		"Sid": pandas.Series([b for a in range(len(TrePlas)) for b in TrePlas[a] * [TrePlansDf.Sid[a]]], dtype=numpy.int32)
		, "PlansIndex": pandas.Series([b for a in TrePlas for b in range(a)], dtype=numpy.int32)
		, "FaPlansIndex": pandas.Series([a - b for a in TrePlas for b in range(a)], dtype=numpy.int32)
		, "Dista":  pandas.Series([a["distance"] for a in TrePlap], dtype=numpy.int32)
		, "Jag":  pandas.Series([a["price"]  if a["price"] != "" else numpy.nan for a in TrePlap]).fillna(0).astype(numpy.int32)
		, "Iota":  pandas.Series([int(a["eta"]) for a in TrePlap], dtype=numpy.int32)
		, "JhMode":  pandas.Series([int(a["transport_mode"]) for a in TrePlap], dtype=numpy.int16)
	}), on="Sid")
	TreModelingPlansDf = pandas.concat([
		CsDf.loc[:, ["Sid", "P", "ChshP", "Qd", "Zd", "QiYoubi", "Qidi", "Qihi"]]
		, XlDf.loc[:, ["Sid", "P", "ChshP", "Qd", "Zd", "QiYoubi", "Qidi", "Qihi"]]
	])
	TreModelingPlansDf["PlansIndex"] = -1
	TreModelingPlansDf["FaPlansIndex"] = -1
	TreModelingPlansDf["Dista"] = -1
	TreModelingPlansDf["Jag"] = -1
	TreModelingPlansDf["Iota"] = -1
	TreModelingPlansDf["JhMode"] = 0
	TreModelingPlansDf = TreModelingPlansDf.astype({"PlansIndex": numpy.int32, "FaPlansIndex": numpy.int32, "Dista": numpy.int32, "Jag": numpy.int32, "Iota": numpy.int32, "JhMode": numpy.int16})
	TrePlansDf = pandas.concat([TrePlansDf, TreModelingPlansDf])
	del TrePlap
	del TrePlas
	del TreModelingPlansDf
	gc.collect()


	def GetKestDataDf(noi, hane = []):
		MTreKestDataDf = TreDf.groupby(noi).agg({"Sid":"count", **{a: "nunique" for a in hane}}).reset_index()
		MTreKestDataDf = MTreKestDataDf.rename({"Sid": "".join(noi) + "NoVonas", **{a: "".join(noi) + "No" + a + "Count" for a in hane}}, axis=1)
		for A in range(1, 12):
			Pref = "".join(noi) + "Mode" + str(A)
			ADf = TrePlansDf[TrePlansDf.JhMode == A].groupby(noi).agg({"Sid": "nunique", **{a: "nunique" for a in hane}}).reset_index()
			ADf.rename({"Sid": Pref + "NoVonas", **{a: Pref + "No" + a + "Count" for a in hane}}, axis=1, inplace=True)
			MTreKestDataDf = MTreKestDataDf.merge(ADf, on=noi, how="left")
			MTreKestDataDf[Pref + "NoVonasbi"] = MTreKestDataDf[Pref + "NoVonas"] / MTreKestDataDf["".join(noi) + "NoVonas"]
		MTreKestDataDf = MTreKestDataDf.fillna(0)
		return MTreKestDataDf

	TreQdKestDataDf = GetKestDataDf(["Qd"], ["ChshP"])
	TreZdKestDataDf = GetKestDataDf(["Zd"], ["ChshP"])
	TreQdZdKestDataDf = GetKestDataDf(["Qd", "Zd"], ["ChshP"])
	TrePKestDataDf = GetKestDataDf(["ChshP"], [])
	TrePQdKestDataDf = GetKestDataDf(["ChshP", "Qd"], [])
	TrePZdKestDataDf = GetKestDataDf(["ChshP", "Zd"], [])
	TrePQdZdKestDataDf = GetKestDataDf(["ChshP", "Qd", "Zd"])

	def GetSestDataDf(noi, hane=[]):
		MTreSestDataDf = TreDf.groupby(noi).agg({"Sid": "count", **{a: "nunique" for a in hane}}).reset_index()
		MTreSestDataDf = MTreSestDataDf.astype({"Sid": numpy.float32, **{a: numpy.float32 for a in hane}})
		MTreSestDataDf.rename({"Sid": "".join(noi) + "NoVonas", **{a: "".join(noi) + "No" + a + "Count" for a in hane}}, axis=1, inplace=True)
		MTreSestDataDf = MTreSestDataDf.fillna(0)
		return MTreSestDataDf

	TreQidiDataDf = GetSestDataDf(["Qidi"], ["ChshP", "Qd", "Zd"])
	TreQdQidiDataDf = GetSestDataDf(["Qd", "Qidi"], ["ChshP", "Zd"])
	TreZdQidiDataDf = GetSestDataDf(["Zd", "Qidi"], ["ChshP", "Qd"])
	TreQdZdQidiDataDf = GetSestDataDf(["Qd", "Zd", "Qidi"], ["ChshP"])
	TrePQidiDataDf = GetSestDataDf(["ChshP", "Qidi"], ["Qd", "Zd"])
	TrePQdQidiDataDf = GetSestDataDf(["ChshP", "Qd", "Qidi"], ["Zd"])
	TrePZdQidiDataDf = GetSestDataDf(["ChshP", "Zd", "Qidi"], ["Qd"])
	TrePQdZdQidiDataDf = GetSestDataDf(["ChshP", "Qd", "Zd", "Qidi"])

	TreQihiDataDf = GetSestDataDf(["Qihi"], ["ChshP", "Qd", "Zd"])
	TreQdQihiDataDf = GetSestDataDf(["Qd", "Qihi"], ["ChshP", "Zd"])
	TreZdQihiDataDf = GetSestDataDf(["Zd", "Qihi"], ["ChshP", "Qd"])
	TreQdZdQihiDataDf = GetSestDataDf(["Qd", "Zd", "Qihi"], ["ChshP"])
	TrePQihiDataDf = GetSestDataDf(["ChshP", "Qihi"], ["Qd", "Zd"])
	TrePQdQihiDataDf = GetSestDataDf(["ChshP", "Qd", "Qihi"], ["Zd"])
	TrePZdQihiDataDf = GetSestDataDf(["ChshP", "Zd", "Qihi"], ["Qd"])
	TrePQdZdQihiDataDf = GetSestDataDf(["ChshP", "Qd", "Zd", "Qihi"])

	def GetAbDataDf(noi, hane = []):
		MTreAbDataDf = TreDf.loc[:, noi + ["Sid", "Qisi", "QdLong", "QdLati", "ZdLong", "ZdLati"]]
		MTreAbDataDf["Quera"] = MTreAbDataDf.groupby(noi).Qisi.rank()

		MTreMaeAbDataDf = MTreAbDataDf.loc[:, noi + ["Qisi", "QdLong", "QdLati", "ZdLong", "ZdLati", "Quera"]]
		MTreMaeAbDataDf.rename({"Qisi": "MaeAbQisi", "QdLong": "MaeAbQdLong", "QdLati": "MaeAbQdLati", "ZdLong": "MaeAbZdLong", "ZdLati": "MaeAbZdLati"}, axis=1, inplace=True)
		MTreMaeAbDataDf["Quera"] = 1 + MTreMaeAbDataDf.Quera
		MTreUshiroAbDataDf = MTreAbDataDf.loc[:, noi + ["Qisi", "QdLong", "QdLati", "ZdLong", "ZdLati", "Quera"]]
		MTreUshiroAbDataDf.rename({"Qisi": "UshiroAbQisi", "QdLong": "UshiroAbQdLong", "QdLati": "UshiroAbQdLati", "ZdLong": "UshiroAbZdLong", "ZdLati": "UshiroAbZdLati"}, axis=1, inplace=True)
		MTreUshiroAbDataDf["Quera"] = MTreUshiroAbDataDf.Quera - 1

		MTreAbDataDf = MTreAbDataDf.merge(MTreMaeAbDataDf, on=noi + ["Quera"], how="left")
		MTreAbDataDf = MTreAbDataDf.merge(MTreUshiroAbDataDf, on=noi + ["Quera"], how="left")
		MTreAbDataDf["".join(noi) + "NoMaeAbQisch"] = abs(MTreAbDataDf.Qisi - MTreAbDataDf.MaeAbQisi).astype(numpy.float32)
		MTreAbDataDf["".join(noi) + "NoUshiroAbQisch"] = abs(MTreAbDataDf.UshiroAbQisi - MTreAbDataDf.Qisi).astype(numpy.float32)

		Exlm = []
		if "Qd" in hane:
			MTreAbDataDf["".join(noi) + "NoMaeAbQdDista"] = pandas.Series([CalUero(Long, Lati, MaeLong, MaeLati) for Long, Lati, MaeLong, MaeLati in zip(MTreAbDataDf.QdLong, MTreAbDataDf.QdLati, MTreAbDataDf.MaeAbQdLong, MTreAbDataDf.MaeAbQdLati)], dtype=numpy.float32)
			MTreAbDataDf["".join(noi) + "NoUshiroAbQdDista"] = pandas.Series([CalUero(Long, Lati, UshiroLong, UshiroLati) for Long, Lati, UshiroLong, UshiroLati in zip(MTreAbDataDf.QdLong, MTreAbDataDf.QdLati, MTreAbDataDf.UshiroAbQdLong, MTreAbDataDf.UshiroAbQdLati)], dtype=numpy.float32)
			MTreAbDataDf["".join(noi) + "NoMaeAbQdDistabisch"] = 	MTreAbDataDf["".join(noi) + "NoMaeAbQdDista"] / MTreAbDataDf["".join(noi) + "NoMaeAbQisch"]
			MTreAbDataDf["".join(noi) + "NoUshiroAbQdDistabisch"] = 	MTreAbDataDf["".join(noi) + "NoUshiroAbQdDista"] / MTreAbDataDf["".join(noi) + "NoUshiroAbQisch"]
			Exlm.extend(["".join(noi) + "NoMaeAbQdDista", "".join(noi) + "NoUshiroAbQdDista", "".join(noi) + "NoMaeAbQdDistabisch", "".join(noi) + "NoUshiroAbQdDistabisch"])

		if "Zd" in hane:
			MTreAbDataDf["".join(noi) + "NoMaeAbZdDista"] = pandas.Series([CalUero(Long, Lati, UshiroLong, UshiroLati) for Long, Lati, UshiroLong, UshiroLati in zip(MTreAbDataDf.ZdLong, MTreAbDataDf.ZdLati, MTreAbDataDf.MaeAbZdLong, MTreAbDataDf.MaeAbZdLati)], dtype=numpy.float32)
			MTreAbDataDf["".join(noi) + "NoUshiroAbZdDista"] = pandas.Series([CalUero(Long, Lati, UshiroLong, UshiroLati) for Long, Lati, UshiroLong, UshiroLati in zip(MTreAbDataDf.ZdLong, MTreAbDataDf.ZdLati, MTreAbDataDf.UshiroAbZdLong, MTreAbDataDf.UshiroAbZdLati)], dtype=numpy.float32)
			MTreAbDataDf["".join(noi) + "NoMaeAbZdDistabisch"] = MTreAbDataDf["".join(noi) + "NoMaeAbZdDista"] / MTreAbDataDf["".join(noi) + "NoMaeAbQisch"]
			MTreAbDataDf["".join(noi) + "NoUshiroAbZdDistabisch"] = MTreAbDataDf["".join(noi) + "NoUshiroAbZdDista"] / MTreAbDataDf["".join(noi) + "NoUshiroAbQisch"]
			Exlm.extend(["".join(noi) + "NoMaeAbZdDista", "".join(noi) + "NoUshiroAbZdDista", "".join(noi) + "NoMaeAbZdDistabisch", "".join(noi) + "NoUshiroAbZdDistabisch"])

		MTreAbDataDf = MTreAbDataDf.loc[:, ["Sid", "".join(noi) + "NoMaeAbQisch", "".join(noi) + "NoUshiroAbQisch"] + Exlm]
		MTreAbDataDf = MTreAbDataDf.fillna(-1)
		return MTreAbDataDf

	TrePAbDataDf = GetAbDataDf(["ChshP"], ["Qd", "Zd"])
	TrePQdAbDataDf = GetAbDataDf(["ChshP", "Qd"], ["Zd"])
	TrePZdAbDataDf = GetAbDataDf(["ChshP", "Zd"], ["Qd"])
	TrePQdZdAbDataDf = GetAbDataDf(["ChshP", "Qd", "Zd"])

	TreSessDataDf = TrePlansDf[TrePlansDf.JhMode != 0].groupby("Sid").agg({"JhMode": "count", "Dista": "min", "Jag": numpy.nanmin, "Iota": "min"}).reset_index()
	TreSessDataDf.rename({"JhMode": "PlanCount", "Dista": "MinDista", "Jag": "MinJag", "Iota": "MinIota"}, axis=1, inplace=True)
	for A in range(1, 12):
		Pref = "Mode" + str(A)
		ADf = TrePlansDf[TrePlansDf.JhMode == A].groupby("Sid").agg({"JhMode": "count", "PlansIndex": "min", "FaPlansIndex": "min", "Dista": "min", "Jag": numpy.nanmin, "Iota": "min"}).reset_index()
		ADf.rename({"JhMode": Pref + "NoPlanCount", "PlansIndex": Pref + "NoMinPlansIndex", "FaPlansIndex": Pref + "NoMinFaPlansIndex", "Dista": Pref + "NoMinDista", "Jag": Pref + "NoMinJag", "Iota": Pref + "NoMinIota"}, axis=1, inplace=True)

		TreSessDataDf = TreSessDataDf.merge(ADf, on="Sid", how="left")
		TreSessDataDf[Pref + "NoPlanCountZhbi"] = TreSessDataDf[Pref + "NoPlanCount"] / (1 + TreSessDataDf.PlanCount)
		TreSessDataDf[Pref + "NoMinDistach"] = TreSessDataDf[Pref + "NoMinDista"] - TreSessDataDf.MinDista
		TreSessDataDf[Pref + "NoMinJagch"] = TreSessDataDf[Pref + "NoMinJag"] - TreSessDataDf.MinJag
		TreSessDataDf[Pref + "NoMinIotach"] = TreSessDataDf[Pref + "NoMinIota"] - TreSessDataDf.MinIota
	for A in [1, 2, 7]:
		Pref = "Mode" + str(A)
		ADf = TrePlansDf[TrePlansDf.JhMode == A].groupby("Sid").agg({"PlansIndex": "max", "FaPlansIndex": "max", "Dista": "max", "Jag": numpy.nanmax, "Iota": "max"}).reset_index()
		ADf.rename({"PlansIndex": Pref + "NoMaxPlansIndex", "FaPlansIndex": Pref + "NoMaxFaPlansIndex", "Dista": Pref + "NoMaxDista", "Jag": Pref + "NoMaxJag", "Iota": Pref + "NoMaxIota"}, axis=1, inplace=True)
		TreSessDataDf = TreSessDataDf.merge(ADf, on="Sid", how="left")
	TreSessDataDf = TreSessDataDf.fillna(-1)

	def GetDataDf(MDf):
		if "DjMode" not in MDf:
			MDf["DjMode"] = numpy.nan
		MDataDf = MDf.loc[:, ["Sid", "DjMode", "P", "ChshP", "Qd", "Zd", "QdLong", "QdLati", "ZdLong", "ZdLati", "QzDista", "Qisi", "Qidi", "Qihi", "Jisi"]]
		MDataDf["Iosta"] = MDataDf.Jisi - MDataDf.Qisi
		MDataDf["Iostb"] = (MDataDf.Qisi - 57600 - 14400) % 86400
		MDataDf["Iostc"] = (MDataDf.Jisi - 57600 - 14400) % 86400

		MDataDf = MDataDf.merge(PDf, on="P", how="left")
		MDataDf = MDataDf.merge(TreQdKestDataDf, on="Qd", how="left")
		MDataDf = MDataDf.merge(TreZdKestDataDf, on="Zd", how="left")
		MDataDf = MDataDf.merge(TreQdZdKestDataDf, on=["Qd", "Zd"], how="left")
		MDataDf = MDataDf.merge(TrePKestDataDf, on="ChshP", how="left")
		MDataDf = MDataDf.merge(TrePQdKestDataDf, on=["ChshP", "Qd"], how="left")
		MDataDf = MDataDf.merge(TrePZdKestDataDf, on=["ChshP", "Zd"], how="left")
		MDataDf = MDataDf.merge(TrePQdZdKestDataDf, on=["ChshP", "Qd", "Zd"], how="left")

		MDataDf = MDataDf.merge(TreQidiDataDf, on=["Qidi"], how="left")
		MDataDf = MDataDf.merge(TreQdQidiDataDf, on=["Qd", "Qidi"], how="left")
		MDataDf = MDataDf.merge(TreZdQidiDataDf, on=["Zd", "Qidi"], how="left")
		MDataDf = MDataDf.merge(TreQdZdQidiDataDf, on=["Qd", "Zd", "Qidi"], how="left")
		MDataDf = MDataDf.merge(TrePQidiDataDf, on=["ChshP", "Qidi"], how="left")
		MDataDf = MDataDf.merge(TrePQdQidiDataDf, on=["ChshP", "Qd", "Qidi"], how="left")
		MDataDf = MDataDf.merge(TrePZdQidiDataDf, on=["ChshP", "Zd", "Qidi"], how="left")
		MDataDf = MDataDf.merge(TrePQdZdQidiDataDf, on=["ChshP", "Qd", "Zd", "Qidi"], how="left")

		MDataDf = MDataDf.merge(TreQihiDataDf, on=["Qihi"], how="left")
		MDataDf = MDataDf.merge(TreQdQihiDataDf, on=["Qd", "Qihi"], how="left")
		MDataDf = MDataDf.merge(TreZdQihiDataDf, on=["Zd", "Qihi"], how="left")
		MDataDf = MDataDf.merge(TreQdZdQihiDataDf, on=["Qd", "Zd", "Qihi"], how="left")
		MDataDf = MDataDf.merge(TrePQihiDataDf, on=["ChshP", "Qihi"], how="left")
		MDataDf = MDataDf.merge(TrePQdQihiDataDf, on=["ChshP", "Qd", "Qihi"], how="left")
		MDataDf = MDataDf.merge(TrePZdQihiDataDf, on=["ChshP", "Zd", "Qihi"], how="left")
		MDataDf = MDataDf.merge(TrePQdZdQihiDataDf, on=["ChshP", "Qd", "Zd", "Qihi"], how="left")

		MDataDf = MDataDf.merge(TrePAbDataDf, on="Sid", how="left")
		MDataDf = MDataDf.merge(TrePQdAbDataDf, on="Sid", how="left")
		MDataDf = MDataDf.merge(TrePZdAbDataDf, on="Sid", how="left")
		MDataDf = MDataDf.merge(TrePQdZdAbDataDf, on="Sid", how="left")

		MDataDf = MDataDf.merge(TreSessDataDf, on="Sid", how="left")
		MDataDf.drop(["P", "ChshP", "Qd", "Zd", "Qidi", "Qihi", "Qisi", "Jisi", "p11", "p18"], axis=1, inplace=True)

		return MDataDf

	CsDataDf = GetDataDf(CsDf)
	XlDataDf = GetDataDf(XlDf)

	del CsDf
	del XlDf
	del TreDf
	del PDf
	del TreQdKestDataDf
	del TreZdKestDataDf
	del TreQdZdKestDataDf
	del TrePKestDataDf
	del TrePQdKestDataDf
	del TrePZdKestDataDf
	del TrePQdZdKestDataDf

	del TreQidiDataDf
	del TreQdQidiDataDf
	del TreZdQidiDataDf
	del TreQdZdQidiDataDf
	del TrePQidiDataDf
	del TrePQdQidiDataDf
	del TrePZdQidiDataDf
	del TrePQdZdQidiDataDf

	del TreQihiDataDf
	del TreQdQihiDataDf
	del TreZdQihiDataDf
	del TreQdZdQihiDataDf
	del TrePQihiDataDf
	del TrePQdQihiDataDf
	del TrePZdQihiDataDf
	del TrePQdZdQihiDataDf

	del TrePAbDataDf
	del TrePQdAbDataDf
	del TrePZdAbDataDf
	del TrePQdZdAbDataDf

	del TreSessDataDf

	for Lm in CsDataDf.columns[16:]:
		if Lm.find("Qidi") >= 0 or Lm.find("Qihi") >= 0:
			CsDataDf.loc[:, [Lm]] = CsDataDf.loc[:, [Lm]] / CsDataDf.loc[:, [Lm]].mean()
			XlDataDf.loc[:, [Lm]] = XlDataDf.loc[:, [Lm]] / XlDataDf.loc[:, [Lm]].mean()

	CsLabelDf = CsDataDf.loc[:, ["Sid"]]
	CsDataDf = CsDataDf.drop(["Sid", "DjMode"], axis=1)
	CsDataDf = CsDataDf.astype({a: numpy.float32 for a, b in zip(CsDataDf.dtypes.index, CsDataDf.dtypes) if b == numpy.float64})
	CsDataDf = CsDataDf.astype({a: numpy.int32 for a, b in zip(CsDataDf.dtypes.index, CsDataDf.dtypes) if b == numpy.int64})
	XlLabelDf = XlDataDf.loc[:, ["Sid", "DjMode"]]
	XlDataDf = XlDataDf.drop(["Sid", "DjMode"], axis=1)
	XlDataDf = XlDataDf.astype({a: numpy.float32 for a, b in zip(XlDataDf.dtypes.index, XlDataDf.dtypes) if b == numpy.float64})
	XlDataDf = XlDataDf.astype({a: numpy.int32 for a, b in zip(XlDataDf.dtypes.index, XlDataDf.dtypes) if b == numpy.int64})

	with open(Prefix + "temp/test_plan", "wb") as file:
		pickle.dump(TrePlansDf.loc[TrePlansDf.Qidi >= 0, ["Sid", "JhMode"]], file)
	with open(Prefix + "temp/test_label", "wb") as file:
		pickle.dump(CsLabelDf, file)
	with open(Prefix + "temp/test_data", "wb") as file:
		pickle.dump(CsDataDf, file)
	del CsLabelDf
	del CsDataDf
	gc.collect()

	for A in range(12):
		XlModeALabelDf = XlLabelDf.merge(TrePlansDf.loc[TrePlansDf.JhMode == A, ["Sid"]].drop_duplicates(), on="Sid", right_index=True)
		XlModeALabelDf["Label"] = (XlModeALabelDf.DjMode == A).astype(numpy.int)

		XlModeADataDf = XlDataDf.loc[XlModeALabelDf.index.tolist()]
		XlModeADataset = lightgbm.Dataset(XlModeADataDf, XlModeALabelDf.Label)
		del XlModeADataDf
		gc.collect()

		Mlgb = lightgbm.train(train_set=XlModeADataset
			, params={"objective": "binary", "learning_rate": 0.04, "max_depth": 6, "num_leaves": 127, "bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 0, "verbose": -1}
			, num_boost_round=800
		)

		with open(Prefix + "model/" + str(A), "wb") as file:
			pickle.dump(Mlgb, file)
		del XlModeADataset
		del XlModeALabelDf

	del XlLabelDf
	del XlDataDf


