import csv

def main():
    with open("data/collocations/2_hashtag_appearances.csv", newline='', encoding="utf-8") as app_orig:
        with open("data/collocations/2_hashtag_collocations.csv", newline='', encoding="utf-8") as coll_orig:
            with open("data/collocations/2_hashtag_appearances_1000plus.csv", "w+", encoding="utf-8") as app_new:
                with open("data/collocations/2_hashtag_collocations_1000plus.csv", "w+", encoding="utf-8") as coll_new:
                    appearances_csv = csv.reader(app_orig, delimiter=' ', quotechar='|')
                    collocations_csv = csv.reader(coll_orig, delimiter=' ', quotechar='|')
                    next(appearances_csv, None)
                    next(collocations_csv, None)
                    app_writer = csv.writer(app_new, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    coll_writer = csv.writer(coll_new, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    app_writer.writerow(["ID", "Appearances"])
                    coll_writer.writerow(["Source", "Target", "Weight"])
                    above_999_set = set()
                    for app_line in appearances_csv:
                        if int(app_line[1]) > 999:
                            above_999_set.add(app_line[0])
                            app_writer.writerow(app_line)
                    #print(above_999_set)
                    check_set = set()
                    for coll_line in collocations_csv:
                        #print(coll_line)
                        if coll_line[0] in above_999_set:
                            if coll_line[1] in above_999_set:
                                coll_writer.writerow(coll_line)

                    print(above_999_set.difference(check_set))
                    print(check_set.difference(above_999_set))
if __name__ == "__main__":
    main()