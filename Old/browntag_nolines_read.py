import nltk

with open("browntag_nolines.txt", "r") as file:
    data = file.read()

sentences = nltk.sent_tokenize(data)

with open("tokenized.txt", "w") as file:
    for sentence in sentences:
        file.write(sentence + "\n")
