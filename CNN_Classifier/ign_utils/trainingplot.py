from tensorflow.python.keras import callbacks
from matplotlib import pyplot as plt

global train_plot_begin_flag
train_plot_begin_flag = 0

global train_next_flag

class PlotLosses(callbacks.Callback):


    def __init__(self):
        global train_plot_begin_flag
        train_plot_begin_flag = 0
    def __del__(self):
        try:
            plt.close()
        except:
            pass


    def on_train_begin(self, logs={}):
        global train_plot_begin_flag
        print("########### train begin ##############")
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.acc = []
        self.val_acc = []

        self.logs = []


        self.plot1 = plt.subplot(2,1,1)
        self.plot2 =  plt.subplot(2,1,2)
        print("train_plot_begin_flag",train_plot_begin_flag)
        if(train_plot_begin_flag == 0):
            #self.close_plt()

            self.plot1 = plt.subplot(2,1,1)
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            self.plot1.plot(self.x, self.acc,color="red", label="training accuracy")
            self.plot1.plot(self.x, self.val_acc,color="black", label="validation accuracy")
            plt.legend(loc=0)
            plt.draw()


            self.plot2 =  plt.subplot(2,1,2)
            plt.xlabel('epochs')
            plt.ylabel('Loss')
            self.plot2.plot(self.x, self.losses,color="red", label="training loss")
            self.plot2.plot(self.x, self.val_losses,color="black", label="validation loss")
            plt.legend(loc=0)
            plt.draw()
            # plt.show(0)
            # plt.pause(0.001)
            train_plot_begin_flag = 1


    def on_epoch_end(self, epoch, logs={}):
        global train_next_flag
        print("########### epoch end ##########3")
        self.i += 1
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        print("logs",logs)


        #plt.subplot(2,1,1)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        self.plot1.plot(self.x, self.acc,color="red")#, label="training accuracy")
        self.plot1.plot(self.x, self.val_acc,color="black")#, label="validation accuracy")
        plt.legend(loc=0)
        plt.draw()


        #plt.subplot(2,1,2)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        self.plot2.plot(self.x, self.losses,color="red")#, label="training loss")
        self.plot2.plot(self.x, self.val_losses,color="black")#, label="validation loss")
        plt.legend(loc=0)
        plt.draw()
        # plt.show(0)
        # plt.pause(0.001)

        #if train_next_flag==1:
         #   print('stopping current training, moving to next model')
         #   self.model.stop_training = True


        #plt.pause(0.001)
        #print("after show")
        # self.fig.canvas.draw()
        #self.fig.clf()
        plt.savefig('output/'+ str(epoch) + '_plot.png')
    def close_plt(self):
        plt.close()
