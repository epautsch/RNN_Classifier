
import pandas

data = pandas.read_csv('training.csv', usecols=['text', 'author'], sep=',')


vocab = set()
for piece in list(data['text']):
    
    vocab.update(piece.split(" "))
    
print(len(vocab))
    
    