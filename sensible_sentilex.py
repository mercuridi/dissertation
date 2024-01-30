
DELIMITER = "|"

with open("SentiLex-PT02/SentiLex-lem-PT02.txt", encoding='utf8') as sentilex_raw:
    sentilex = []
    for raw_line in sentilex_raw:
        data_list = []
        raw_line = raw_line.strip()
        lemma, _, remainder = raw_line.partition(".")
        data_list.append(lemma)
        data = remainder.split(";")
        for datapoint in data:
            key, value = datapoint.split("=")
            data_list.append(value)
        sentilex.append(data_list)

    with open("data/sentilex.txt", encoding="utf8", mode = "w") as processed:
        processed.write("POS"       + DELIMITER + \
                        "TG"        + DELIMITER + \
                        "POL:N0"    + DELIMITER + \
                        "POL:N1"    + "\n")
        for item in sentilex:
            tg = item[2]
            write_this = ""
            match tg:
                case "HUM:N0":
                    for i in range(0, 5):
                        if i == 4:
                            write_this += "NaN"
                            write_this += DELIMITER
                            break
                        write_this += item[i]
                        write_this += DELIMITER
                case "HUM:N1":
                    for i in range(0, 4):
                        if i == 3:
                            write_this += "NaN"
                            write_this += DELIMITER
                        write_this += item[i]
                        write_this += DELIMITER
                case "HUM:N0:N1":
                    for i in range(0, 5):
                        write_this += item[i]
                        write_this += DELIMITER
                case other:
                    print(f"FATAL ERROR IDENTIFYING TARGET {tg}")
                    exit()


            write_this = write_this[:len(write_this)-1]
            processed.write(write_this+"\n")
    
print("Processing finished. Written to sentilex.txt")