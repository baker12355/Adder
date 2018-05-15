from keras.models import load_model

expected=[]
thefile = open('expected.txt', 'r')
for i in thefile:
    expected.append(i)

questions=[]
thefile = open('questions.txt', 'r')
for i in thefile:
    questions.append(i)


model = load_model('model.h5')


