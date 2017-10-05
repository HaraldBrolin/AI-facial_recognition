#import numpy
import random
import math

class Perceptron:

    path_to_training = r"H:\Desktop\AI\AI_2\Test_set\training-A.txt"
    path_to_ans = r"H:\Desktop\AI\AI_2\Test_set\facit-A.txt"

    def __init__(self):
        self.training_set = training_to_dic()
        self.training_ans = training_ans_to_dict()

    def read_file(self, file_path):
        with open(file_path) as file:
            data_set = file.readlines()
            file.close()
        return data_set

    def training_to_dic(self):
        training_set = self.read_file(Perceptron.path_to_training)
        list_of_img = []
        list_of_key = []
        for row, line in enumerate(training_set):
            if line.startswith("Image") is True:
                list_of_key.append(line.rstrip())
                img = []
                for numbers in training_set[row + 1:row + 21]:
                    numbers = numbers.strip()
                    # numbers = [int(i)/31 for i in numbers.split()]
                    numbers = [int(i) / 31 for i in
                               numbers.split()]  # Signal goes from -1 to 1, this is since every function given needs -1 to 1
                    img += numbers
                list_of_img.append(img)
        d = {}
        for index, key in enumerate(list_of_key):
            d[key] = list_of_img[index]
        return d

    def training_ans_to_dict(self):
        training_ans = self.read_file(Perceptron.path_to_ans)
        list_of_ans = []
        list_of_key = []
        for line in training_ans:
            if line.startswith("Image") is True:
                list_of_key.append(line.split()[0])
                if int(line.split()[1]) == 1 or int(line.split()[1]) == 2:
                    list_of_ans.append(1)
                else:
                    list_of_ans.append(0)
                    #list_of_ans.append(line.split()[1])
        d = {}
        for index, key in enumerate(list_of_key):
            d[key] = list_of_ans[index]
        return d




def activation(image, weights):  # Takes a randomly generated list of weights and multiplies by signal for all pixels
    activ = weights[0]
    for index, pixel in enumerate(image):
        # print(index)
        activ += weights[index + 1] * pixel
    # return 1.0 if activ >= 1.0 else 0.0
    return (1 / (1 + math.pow(math.e, -activ)))


def training_to_dic(training_set):
    list_of_img = []
    list_of_key = []
    for row, line in enumerate(training_set):
        if line.startswith("Image") is True:
            list_of_key.append(line.rstrip())
            img = []
            for numbers in training_set[row + 1:row + 21]:
                numbers = numbers.strip()
                # numbers = [int(i)/31 for i in numbers.split()]
                numbers = [int(i) / 31 for i in numbers.split()]  # Signal goes from -1 to 1, this is since every function given needs -1 to 1
                img += numbers
            list_of_img.append(img)
    d = {}
    for index, key in enumerate(list_of_key):
        d[key] = list_of_img[index]
    return d


def change_weights(error, weights, image):
    learning_rate = 0.1
    for index, weight in enumerate(weights[1:]):
        weights[index] = weight + (learning_rate * error * image[index])
    return weights


weights = [0 for x in range(401)]  # Create list of random numbers from 0 to 1
answer = training_ans_to_dict(read_file(r"H:\Desktop\AI\AI_2\Test_set\facit-A.txt"))
training_set = training_to_dic(read_file(r"H:\Desktop\AI\AI_2\Test_set\training-A.txt"))

accuracy = []

for index, tuple in enumerate(answer.items()):
    correct = tuple[1]
    prediction = activation(training_set[tuple[0]], weights)
    #print(prediction)
    error = correct - prediction
    weights = change_weights(error, weights, training_set[tuple[0]])
    if prediction >= 0.5:
        print(correct - 1)
    else:
        print(correct)
    #print(error)



# activ = activation(training_set["Image1"], weights)
# error = answer["Image1"] - activ
#
# change_weights(error, weights, training_set["Image1"])
#print(error)

# training_set = read_file(r"H:\Desktop\AI\AI_2\Test_set\training-A.txt")
# images = file_to_dic(training_set)
# print(activation(images["Image1"], weights))

# print(images["Image1"])
# print(len(images["Image2"]))




# # Make a prediction with weights
# def predict(row, weights):
#     activation = weights[0]
#     for i in range(len(row) - 1):
#         activation += weights[i + 1] * row[i]
#     return 1.0 if activation >= 0.0 else 0.0
#
#
# # test predictions
# dataset = [[2.7810836, 2.550537003, 0],
#            [1.465489372, 2.362125076, 0],
#            [3.396561688, 4.400293529, 0],
#            [1.38807019, 1.850220317, 0],
#            [3.06407232, 3.005305973, 0],
#            [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1],
#            [6.922596716, 1.77106367, 1],
#            [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]]
# weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
# for row in dataset:
#     prediction = predict(row, weights)