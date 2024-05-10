import matplotlib.pyplot as plt

class modelplot(object):
    
        
    def trainPlot(loss:dict,savefig_path):
        for i in loss.keys():
            plt.plot(loss[i],'-')
        plt.legend(loss.keys())
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(savefig_path)
        plt.close()