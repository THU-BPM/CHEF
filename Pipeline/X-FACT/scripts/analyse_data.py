import sys
import pprint

if __name__=="__main__":

    filename = sys.argv[1]


    label_count = {}


    with open(filename, 'r') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            lang = arr[1].lower()
            site = arr[2].lower()
            domain = (lang, site)

            if domain not in label_count:
                label_count[domain] = {}

            label = arr[-1].lower()

            if label not in label_count[domain]:
                label_count[domain][label] = 0
            label_count[domain][label] +=1


    pprint.pprint(label_count)


    new_map = {}
    for key in label_count.keys():
        new_map[key] = {}
        counts = label_count[key]

        total = 0
        for k, v in counts.items():
            total += v

        for k,v in counts.items():
            new_map[key][k] = float(v)/total


    pprint.pprint(new_map)

    total = 0
    count = 0
    for key in new_map.keys():
        counts = new_map[key]
        take = True
        for k, v in counts.items():
            if v > 0.7:
                take = False

        if take:
            print(key)
            count +=1
            for k, v in label_count[key].items():
                total += v

    print(total)
    print(count)
