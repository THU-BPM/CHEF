import sys, os, random
# Remember to only randomize snippets in train set

random.seed(1)


def load_examples(filename):
    snippets_dict = {}
    examples = []

    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i ==0:
                header = line.strip()
                continue
            arr = line.strip().split('\t')
            lang = arr[0]
            site = arr[1]
            examples.append(arr)
            if (lang, site) not in snippets_dict:
                snippets_dict[(lang, site)] = []
            snippets_dict[(lang, site)].append(arr[2:-2])

    return examples, snippets_dict, header


if __name__=="__main__":

    inp = sys.argv[1]
    examples, snippets_dict, header = load_examples(inp)

    out = sys.argv[2]
    new_examples = []
    for ex in examples:
        lang = ex[0]
        site = ex[1]
        random_snippet = random.choice(snippets_dict[(lang, site)])
        new_ex = ex[0:2] + random_snippet + ex[-2:]
        new_examples.append(new_ex)

    with open(out, 'w') as fp:
        fp.write(header + '\n')
        for ex in new_examples:
            fp.write('\t'.join(ex) + '\n')
