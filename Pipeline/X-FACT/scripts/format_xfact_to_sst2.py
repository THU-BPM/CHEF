import sys

def load_examples(filename):
    examples = []

    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            sentence = line.strip().split('\t')[-2]
            label = line.strip().split('\t')[-1]

            examples.append((sentence, label))

    return examples

if __name__=="__main__":

    inp = sys.argv[1]
    out = sys.argv[2]

    examples = load_examples(inp)

    fp = open(out, 'w')

    fp.write('sentence\tlabel\n')
    for ex in examples:
        fp.write(ex[0] + '\t' + ex[1] + '\n')

    fp.close()
