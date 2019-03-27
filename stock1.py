from keras.models import  Sequential
import keras.layers as kl
from keras.layers import Dense
from keras.models import Model
from keras import regularizers
import keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, figure, show
import csv

from keras.models import load_model

from keras.layers.recurrent import LSTM
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import GridSearchCV


class NeuralNetwork:
    def __init__(self, input_shape, stock_or_return):
        self.input_shape = input_shape
        self.stock_or_return = stock_or_return

    def Adjust_params(self, train_model, train, train_y):
        print(train.shape)
        model = KerasClassifier(build_fn=train_model,  epochs=50, batch_size=40, shuffle=True)
        print("params:",model.get_params())
        dic = {"epochs":range(0,100,10),"batch_size":range(0,100,10),"shuffle":[True,False]}
        grid = GridSearchCV(estimator=model, param_grid=dic, n_jobs=1)
        grid = grid.fit(train, train_y, dic)
        print(grid.best_params_)

    def BestModel(self, train, train_y, test):
        activation = ["softmax","elu","selu","softplus","softsign","relu","tanh","sigmoid","hard_sigmoid","exponential","linear"]
        Path = []
        for a in activation:
            filepath = "ourmodels/%s/BestModel_%s.h5" % ("activation",a)
            Path.append(filepath)
            print("--------------------------------------------------------------------------------------------------------------------")
            print("activation:", a)
            input_data = kl.Input(shape=(10, self.input_shape))
            # lstm = kl.LSTM(10,input_shape=(10, self.input_shape),activity_regularizer=regularizers.l2(0.003),
            #                 recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
            
            lstm = kl.LSTM(10, input_shape=(10, self.input_shape), activation=a)(input_data)

            perc = kl.Dense(1,input_shape=(10, self.input_shape))(lstm)
           # perc = kl.Dense(1,input_shape=(10, self.input_shape))(lstm)

            model=Model(input_data,perc)

            checkpoint = ModelCheckpoint(filepath = filepath,
                                    monitor = "loss",
                                    verbose = 1,
                                    save_best_only='True',
                                    mode='auto'
                                    )

            callback_lists = [checkpoint]
            model.compile(optimizer="adam", loss="mean_squared_error",metrics=['mse'])
            model.fit(train, train_y, epochs=50,batch_size=40,shuffle=True,validation_split=1, verbose=1, callbacks=callback_lists)
        L = []
        for path in Path:
            tmpmodel = load_model(path)
            loss, acc = tmpmodel.evaluate(train, train_y)
            L.append(loss)

        finalpath = Path[L.index(min(L))]
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print("finalpath", finalpath)
        print(L.index(min(L)))

        model = load_model(finalpath)


        # model.fit(train, train_y, epochs=50,batch_size=i,shuffle=False)
        # filepath = "ourmodels/batch_model_%d.h5" % (i)
        # model.save(filepath, overwrite=True, include_optimizer=True)
        # model.fit(train, train_y, epochs=50,batch_size=40,shuffle=True,validation_split=1, verbose=1, callbacks=callback_lists)
        # model = load_model(filepath)

        return model


    def make_train_model(self):
        pd_train = pd.read_csv("train_ma.csv")
        pd_train_y = pd.read_csv("train_y.csv")


    

        # load data
        


        #print(pd_train)
        train = np.reshape(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,:].values),
                           (len(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,:].values))//10, 10, self.input_shape))
        print(pd_train_y.shape[0]//10*10)
        train_y = np.array(pd_train_y.iloc[:pd_train_y.shape[0]//10*10,:].values)
        # train_stock = np.array(pd.read_csv("train_stock.csv"))
        train_y=train_y
        train=train
        # train model
        print(train_y.shape)
        #train_y=np.reshape(train_y,(1,train_y.shape[0],train_y.shape[1]))

        # self.Adjust_params(model, train, train_y)

        dic = {"epochs":range(0,100,10),"batch_size":40,"shuffle":[True]}
        # model.fit(train, train_y, epochs=50,batch_size=10,shuffle=False)

        key = "validation_split"

        filepath = "best_model%s.h5" % key

        pd_test = pd.read_csv("test_ma.csv")
        test=np.array(pd_test.iloc[:,:].values)

        test_x = np.reshape(test,
                            (len(test)//10, 10, self.input_shape))

        model = self.BestModel(train, train_y, test_x)


   
        prediction_data = []
        stock_data = []
        for i in range(len(test_x)):
            prediction = (model.predict(np.reshape(test_x[i], (1, 10, self.input_shape))))
         
            prediction_data.append(np.reshape(prediction, (1,)))
            prediction_corrected = (prediction_data - np.mean(prediction_data))/np.std(prediction_data)
        caseid=143
        test_f=open("test_data.csv")
        df_test=pd.read_csv(test_f)
        temppanda=df_test.iloc[1420:,:]
        std_data=temppanda.iloc[:,3:10].values
        ind=10
        with open('write.csv','w',newline='') as csv_file:
          csv_writer = csv.writer(csv_file)
          csv_writer.writerow(["caseid","midprice"])
          for price in prediction_data:
            li=[]
            li.append(caseid)
            li.extend(price+std_data[ind-1,0])
            csv_writer.writerow(li)
            caseid+=1
            ind+=10
        print(len(prediction_data))
        return model



if __name__ == "__main__":
    model = NeuralNetwork(7, True)
    model.make_train_model()
