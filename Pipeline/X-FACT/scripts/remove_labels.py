import sys

def load_examples(filename, labels_to_remove):

    examples = []
    header = None
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                header = line.strip()
                continue
            label = line.strip().split('\t')[-1].strip()
            if label in labels_to_remove:
                continue
            examples.append(line.strip())

    return examples, header

def write_examples(examples, header, filename):

    fp = open(filename, 'w')
    fp.write(header + '\n')

    for ex in examples:
        fp.write(ex + '\n')

    fp.close()

if __name__=="__main__":

    labels_to_remove = ['complicated/hard to categorise', 'other']


    examples, header = load_examples(sys.argv[1], labels_to_remove)

    write_examples(examples, header, sys.argv[2])
