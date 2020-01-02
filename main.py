#!/usr/bin/env python3

import json
import signal

from complexthinking import NeuralNetwork as ComplexNeurones
from simplethinking import NeuralNetwork as SimpleNeurone

MATPLOT = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('install matplotlib via your package manager')
    MATPLOT = False


global FULLDATA, QUESTIONS, NeuralNetwork, NAMES, INPUTS, OUTPUTS, NN


def test_or_train():
    print("Answer must be beetween 0 and 10\n0 stands for \"Absolutely not\"\n10 stands for \"Hell Yeah!\"")
    try:
        name = input('Name of the subject : ')
    except EOFError:
        return
    data = []
    for question in QUESTIONS:
        answer = -1
        while answer > 10 or answer < 0:
            print(question, '?', '(', name, ')')
            try:
                answer = float(input('Truthyness (beetween 0 and 10) : '))
            except ValueError:
                continue
        data.append(answer / 10)
    example = [data]
    prediction = NN.predict(example)[0][0]
    print('Prediction for', name, 'is :', "Oui" if prediction > 0.5 else "Non", '(', prediction, ')')
    correct = None
    while correct not in ['n', 'y']:
        correct = input('Improve answers (y/n) : ')
    if correct == 'n':
        return
    try:
        rate = float(input("How much would you rate it (0 to 10) ? "))
    except:
        return
    NAMES.append(name)
    INPUTS.append(data)
    OUTPUTS.append([rate / 10])
    if INPUTS and not len(INPUTS) % 5:
        data_json = [NAMES, INPUTS, OUTPUTS]
        with open('tmp.json', 'w') as f:
            f.write(json.dumps(data_json))
        NN.update_data(INPUTS, OUTPUTS)
        show_graph()


def add_question():
    nquestion = input("Assertion : ")
    QUESTIONS.append(nquestion)
    for j in range(len(INPUTS)):
        print("Answer must be beetween 0 and 10\n0 stands for \"Absolutely not\"\n10 stands for \"Hell Yeah!\"")
        name = NAMES[j]
        answer = -1
        while answer > 10 or answer < 0:
            print(nquestion, '?', '(', name, ')')
            try:
                answer = float(input('Truthyness : '))
            except ValueError:
                continue
        INPUTS[j].append(answer / 10)
    NN.update_data(INPUTS, OUTPUTS)

    check_data()

    save_data()


def correct_scores():
    for j in range(len(OUTPUTS)):
        print("Answer must be beetween 0 and 10\n0 stands for \"Absolutely not\"\n10 stands for \"Hell Yeah!\"")
        name = NAMES[j]
        prediction = NN.predict([INPUTS[j]])[0][0]
        print('Prediction for', name, 'is :', round(prediction, 2) * 10)
        print('Last rate was :', OUTPUTS[j][0] * 10)
        try:
            rate = float(input("How much would you rate it (0 to 10) ? "))
        except EOFError:
            print("Skipping", name)
            continue
        OUTPUTS[j] = [round(rate / 10, 3)]

    NN.update_data(INPUTS, OUTPUTS)

    save_data()


def correct_data():
    for j in range(len(INPUTS)):
        print("Answer must be beetween 0 and 10\n0 stands for \"Absolutely not\"\n10 stands for \"Hell Yeah!\"")
        try:
            name = input('Name of the subject (old name was \"{}\") : '.format(NAMES[j])) or NAMES[j]
        except EOFError:
            print("Skipping", NAMES[j])
            continue
        data = []
        for i in range(len(QUESTIONS)):
            answer = -1
            while answer > 10 or answer < 0:
                print(QUESTIONS[i], '?', '(', name, ')')
                try:
                    answer = float(input('Truthyness (last was {}) : '.format(INPUTS[j][i] * 10)))
                except ValueError:
                    answer = INPUTS[j][i] * 10
            data.append(answer / 10)
        example = [data]
        prediction = NN.predict(example)[0][0]
        print('Prediction for', name, 'is :', "Oui" if prediction > 0.5 else "Non", '(', prediction, ')')
        rate = float(input("How much would you rate it (0 to 10) ? "))
        NAMES[j] = name
        INPUTS[j] = data
        OUTPUTS[j] = [rate / 10]

    NN.update_data(INPUTS, OUTPUTS)

    save_data()


def correct(name):
    if name not in NAMES:
        print('"{}" Not found'.format(name))
        return
    j = NAMES.index(name)
    print("Answer must be beetween 0 and 10\n0 stands for \"Absolutely not\"\n10 stands for \"Hell Yeah!\"")
    try:
        name = input('Name of the subject (old name was \"{}\") : '.format(NAMES[j])) or NAMES[j]
    except EOFError:
        print("Aborting")
        return
    data = []
    for i in range(len(QUESTIONS)):
        answer = -1
        while answer > 10 or answer < 0:
            print(QUESTIONS[i], '?', '(', name, ')')
            try:
                answer = float(input('Truthyness (last was {}) : '.format(INPUTS[j][i] * 10)))
            except ValueError:
                answer = INPUTS[j][i] * 10
        data.append(answer / 10)
    example = [data]
    prediction = NN.predict(example)[0][0]
    print('Prediction for', name, 'is :', "Oui" if prediction > 0.5 else "Non", '(', prediction, ')')
    rate = float(input("How much would you rate it (old was {})? ".format(OUTPUTS[j][0] * 10)))
    NAMES[j] = name
    INPUTS[j] = data
    OUTPUTS[j] = [rate / 10]

    NN.update_data(INPUTS, OUTPUTS)

    save_data()


def check_data():
    classment = []
    for j in range(len(INPUTS)):
        example = [INPUTS[j]]
        prediction, scores = NN.predict(example, True)
        prediction = round(prediction[0][0], 2)
        scores = list(round(x, 2) for x in scores[0])
        classment.append((prediction, NAMES[j]))
        print('Prediction for', NAMES[j], 'is :', prediction, 'vs', OUTPUTS[j], '(', scores, ')')

    classment.sort(key=lambda x: -x[0])

    print("\n---------------------------\n")

    print("Highest to lowest values :")
    for i in range(1, len(classment) + 1):
        print("#{} : {} ({})".format(i, classment[i - 1][1], classment[i - 1][0]))

    show_graph()
    print("\n---------------------------\n")


def handle_ipdb(sig, frame):
    import ipdb
    ipdb.set_trace(frame)


def get_complex():
    complexity = input("Complexity (0-1) : ")
    print('You choose :', complexity)
    if complexity == '1':
        nweight = int(input("How many weights ? "))
        weights = []
        for x in range(nweight - 1):
            weights.append(int(input("Ouputs of weight {} : ".format(x + 1))))
        ComplexNeurones.weights_config = weights
        return ComplexNeurones
    else:
        return SimpleNeurone


def save_data():
    save = input("Save Datas ? (y/N) ")
    if save == "y":
        FULLDATA[CHOOSENKEY] = [QUESTIONS, NAMES, INPUTS, OUTPUTS]
        data_json = json.dumps(FULLDATA)
        with open('training.json', 'w') as f:
            f.write(data_json)
        print("Saved")


def show_graph():
    if MATPLOT:
        # plot the error over the entire training duration
        plt.figure(figsize=(15, 5))
        plt.plot(NN.epoch_list, NN.error_history)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()


def questions_from_file(attempt=False):
    fileName = input("Absolute path for file: ")
    try:
        f = open(fileName)
    except IOError:
        print("Wrong File Name \"{}\"".format(fileName))
        if attempt:
            raise
        return questions_from_file(True)
    else:
        questions = []
        for line in f:
            questions.append(line.strip())
        f.close()
    return questions


def questions_from_input():
    print("Assertions (max 100):")

    questions = []
    for i in range(1, 101):
        try:
            questions.append(input("#{} Assertion (must be answerable beetween 0 and 10 : ".format(i)))
        except EOFError:
            break
    else:
        print("Too many questions, max is 100")

    return questions


def data_from_file(attempt=False):
    fileName = input("Absolute path for file: ")
    try:
        f = open(fileName)
    except IOError:
        print("Wrong File Name \"{}\"".format(fileName))
        if attempt:
            raise
        return data_from_file(True)
    else:
        names = []
        inputs = []
        outputs = []
        data_json = json.loads(f.read())
        for data in data_json:
            names.append(data['name'])
            inputs.append(list(x / 10 for x in data['input']))
            outputs.append([data['output'] / 10])
        f.close()

    return names, inputs, outputs


def data_from_input():
    names = []
    inputs = []
    outputs = []
    for _ in range(100):
        try:
            name = input("Name : ")
        except EOFError:
            break
        data = []
        for question in QUESTIONS:
            print(question, " ? ")
            data.append(float(input("Answer (0-10): ")) / 10)
        out = float(input("Note that should be given (0-10): "))
        names.append(name)
        inputs.append(data)
        outputs.append([out / 10])
    else:
        print("Too many inputs, max is 100")

    return names, inputs, outputs


def new_test():
    simuName = input("Simulation name: ")
    qFile = input("Use file for assertions \"{}\" ? (y/N) ".format(simuName))
    if qFile == 'y':
        questions = questions_from_file()
    else:
        questions = questions_from_input()

    aFile = input("""
        File should be a json formatted as follow:\n
        [{\n
            "name": "",\n
            "input": [],\n
            "output": 0-10\n
        },\n
        ...])\n
        Use file for datas ? (y/N) """)

    if aFile == 'y':
        names, inputs, outputs = data_from_file()
    else:
        names, inputs, outputs = data_from_input()

    FULLDATA[simuName] = [questions, names, inputs, outputs]

    return simuName


def get_next_step():
    while True:
        try:
            nextstep = input('Test / Correct / Complete / Change Complexity / Exit ? ')
        except EOFError:
            nextstep = None
        if nextstep is None or nextstep.lower() == "exit":
            raise StopIteration
        yield nextstep


def prepare_data(keys):
    inputData = ""
    for x, test in zip(range(len(keys)), keys):
        inputData += "[{}] {}\n".format(x, test)
    inputData += "[{}] {}\n".format(x + 1, "New")

    return inputData


def choose_test():
    keys = list(FULLDATA.keys())
    inputData = prepare_data(keys)

    print("Choose a test :")
    print(inputData)
    choosen_test = int(input("Choice : "))
    if choosen_test == len(keys):
        print("Creating a new test...")
        return new_test()
    else:
        print("Starting {}...".format(keys[choosen_test]))
        return keys[choosen_test]


def main(nextstep):
    global NN, QUESTIONS, NeuralNetwork

    if nextstep == 'correct':
        print("Correction starts")
        correction = input("Correct \"all\", \"<user>\", or \"scores\" ? ")
        if correction == "all":
            correct_data()
        elif correction == "scores":
            correct_scores()
        else:
            correct(correction)
    elif nextstep == 'complete':
        print("Adding an assertion")
        add_question()
    elif nextstep == 'check':
        NN.get_questions_weights(QUESTIONS)
        print("Checking datas")
        check_data()
    elif nextstep in ['cc', 'change complexity']:
        NeuralNetwork = get_complex()
        NN = NeuralNetwork(INPUTS, OUTPUTS)
    else:
        print("Testing starts")
        test_or_train()


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, handle_ipdb)

    with open('training.json') as f:
        FULLDATA = json.loads(f.read())

    CHOOSENKEY = choose_test()
    QUESTIONS, NAMES, INPUTS, OUTPUTS = FULLDATA[CHOOSENKEY]

    NeuralNetwork = get_complex()
    NN = NeuralNetwork(INPUTS, OUTPUTS)

    for nextstep in get_next_step():
        try:
            main(nextstep.lower())
        except (EOFError, KeyboardInterrupt, ValueError) as e:
            from traceback import print_exc
            print("We encountered an unexpected behavior :", e)
            print_exc()
            ret = input("Exit ? Y/n")
            if ret != 'n':
                break
    else:
        save_data()
        show_graph()
