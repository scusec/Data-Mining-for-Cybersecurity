import pickle


def process_dga():
    with open("dga.txt", "r") as f_in:
        res = []
        for line in f_in:
            temp = line.split("\t")[1]
            temp = temp.split(".")[0]
            res.append(temp)
        pickle.dump(res, open("dga_processed.pickle", "wb"))


def process_non_dga():
    with open("top-1m.csv", "r") as f_in:
        res = []
        for line in f_in:
            temp = line.split(".")[0]
            res.append(temp)
        pickle.dump(res, open("non_dga_processed.pickle", "wb"))


if __name__=="__main__":
    process_dga()
    process_non_dga()
