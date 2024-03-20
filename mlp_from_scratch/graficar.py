import matplotlib.pyplot as plt
def graficar_pr(recall_list,precision_list):
        plt.style.use('rose-pine')
        plt.plot(recall_list,precision_list,color='#fb9f9f',marker='*')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.show()

def graficar_rc(loss_list):
    plt.style.use('rose-pine')
    plt.plot(loss_list,color='#fb9f9f')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Rate of Convergency')
    plt.show()

if __name__=="__MAIN__":
      graficar_pr()
      graficar_rc()