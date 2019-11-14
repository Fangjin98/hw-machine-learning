import numpy as np
import matplotlib.pyplot as plt
from utils.data_processor import load_res_data

def plt_space_loss():
    x = np.linspace(0, 1000, 100)
    bp_20_5 = load_res_data('res/bp_20_5')
    bp_20_3 = load_res_data('res/bp_20_3')
    bp_20_1 = load_res_data('res/bp_20_1')
    plt.figure()
    plt.title('study rate -- loss')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.plot(x, bp_20_5,label='study_rate=0.5, hidden_num=20')
    plt.plot(x, bp_20_3,label='study_rate=0.3, hidden_num=20')
    plt.plot(x, bp_20_1,label='study_rate=0.1, hidden_num=20')
    plt.legend()
    plt.savefig('rate_against_loss.jpg')
    plt.show()

def plt_hiddennum_loss():
    x = np.linspace(0, 1000, 100)
    bp_5_1 = load_res_data('res/bp_5_1')
    bp_10_1 = load_res_data('res/bp_10_1')
    bp_20_1 = load_res_data('res/bp_20_1')
    bp_5_3 = load_res_data('res/bp_5_3')
    bp_10_3 = load_res_data('res/bp_10_3')
    bp_20_3 = load_res_data('res/bp_20_3')
    bp_5_5 = load_res_data('res/bp_5_5')
    bp_10_5 = load_res_data('res/bp_10_5')
    bp_20_5 = load_res_data('res/bp_20_5')
    plt.figure(figsize=(10,10))
    plt.title('hidden num -- loss')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.plot(x, bp_5_1,color='b',label='study_rate=0.1, hidden_num=5')
    plt.plot(x, bp_5_3, 'b--', label='study_rate=0.3, hidden_num=5')
    plt.plot(x, bp_5_5, 'b-*', label='study_rate=0.5, hidden_num=5')
    plt.plot(x, bp_10_1,color='r',label='study_rate=0.1, hidden_num=10')
    plt.plot(x, bp_10_3,'r--', label='study_rate=0.3, hidden_num=10')
    plt.plot(x, bp_10_5, 'r-*', label='study_rate=0.5, hidden_num=10')
    plt.plot(x, bp_20_1, color='g', label='study_rate=0.1, hidden_num=20')
    plt.plot(x, bp_20_3, 'g--',label='study_rate=0.3, hidden_num=20')
    plt.plot(x, bp_20_5, 'g-*', label='study_rate=0.5, hidden_num=20')
    plt.legend()
    plt.savefig('hidden_against_loss.jpg')
    plt.show()

def plt_hiddennum_accurcy():
    accurcy_1 = [0.9594, 0.9265, 0.9129]
    loss_1 = [14.4632, 21.5835, 26.1103]
    accurcy_2=[0.9613,0.2437,0.8665]
    loss_2=[12.4020,20.2490,21.7016]
    plt.figure()
    plt.title('study rate -- loss')
    plt.xlabel('accuracy')
    plt.ylabel('loss')
    plt.plot(accurcy_1,loss_1,'ro')
    plt.plot(accurcy_2, loss_2, 'bo')
    plt.savefig('accuracy_against_batch.jpg')
    plt.show()
def main():
    #plt_space_loss()
    #plt_hiddennum_loss()
    plt_hiddennum_accurcy()
if __name__ == '__main__':
    main()
