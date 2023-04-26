import matplotlib.pyplot as plt
import numpy as np

training_loss_exp_2 = np.array([
    1.3774412908892293, 1.1909278652090936, 1.0950712974647152,
    1.030518825845091, 0.9242444450681818, 0.8174636854644964,
    0.6771172991315135, 0.5658861032684728, 0.44907511954878454,
    0.3714316756700791, 0.31664149474341846, 0.27960762781330933,
    0.24579022553982674, 0.22186044827174073, 0.20025163803796472,
    0.18462793026989574, 0.18420013912928418, 0.16669920430361052,
    0.15031527743781428, 0.1532488346942938, 0.1321599065223351, 
    0.14669550881201412, 0.12978243023687985, 0.11451477298022994, 
    0.111347312777524, 0.11496065464059617, 0.1088107303821282,
    0.09633659294444041, 0.10924031607784686, 0.10665651918008195,
    0.08820984757267099, 0.08671609968607462, 0.07024351756385391,
    0.09965716188877305, 0.07182269039188563, 0.06720618315301745,
    0.08458772388929223, 0.07118062453424241, 0.07514930107307426,
    0.06341707841407955
])

evaluation_loss_exp_2 = np.array([
    1.3209326661295362, 1.1812481801588457, 1.297786375978491, 
    1.2462230492173956, 1.2777704167497026, 1.3606191375132264,
    1.4773594050229282, 1.6105404269364145, 1.8878512591466137,
    2.011748807668522, 2.1657727608624637, 2.060134145584735, 
    2.5070049755952577, 2.569680387518187, 2.6993623510511084,
    3.0078342270519998, 3.10539235861765, 3.013831274376975, 
    3.2667708694934845, 3.317839024082194, 3.412585671028394, 
    2.935189796590257, 2.8416285336493177, 3.213794700215372, 
    3.3525958739811297, 3.7004942523053614, 3.576920302932734, 
    3.7360175975253074, 3.5093558466304904, 3.9640411665273265,
    3.2931112392442836, 3.4352910132302594, 3.8733828206421297,
    3.506162358986007, 4.035775781933511, 3.8859656677496703,
    3.007547470910305, 4.083014329605632, 3.840607860849963, 
    3.9828260096391057
])

training_acc_exp_2 = np.array([
    0.5404970736077607, 0.6006994067503622, 0.6362885653284893, 
    0.6472099065968798, 0.6828585381158443, 0.7174548483483845,
    0.7613730399512842, 0.8043803222121522, 0.8448496140719063, 
    0.871869366265918, 0.8896972065443403, 0.9029788779946409, 
    0.9138510514457484, 0.9248538802969183, 0.9298842706888684,
    0.9387216483699373, 0.9378564934925137, 0.9418716253922247, 
    0.9510354915811768, 0.9475282910660269, 0.955843479612419, 
    0.9499911262820583, 0.9572243100980539, 0.9617317449005347, 
    0.9615507328629934, 0.9594334217295322, 0.9644316992531278, 
    0.9654432404272353, 0.9625707815068857, 0.9658626052162174, 
    0.9693866359640919, 0.9711511802803937, 0.9769547590907744,
    0.9674175170415772, 0.9747930333907491, 0.9773112308781099,
    0.9746686904088429, 0.9773822787204399, 0.977008740354605, 
    0.9797086746866344
])

evaluation_acc_exp_2 = np.array([
    0.5523323744157077, 0.6041836978650166, 0.5717185592185592, 
    0.5956032419515566, 0.5871063612821854, 0.6042792616068479, 
    0.5579169234341648, 0.5594102964936298, 0.5693456766732629, 
    0.5765753324819259, 0.5434257845774699, 0.510273990356408,
    0.5304197380901926, 0.5584056109918178, 0.5498032586100768, 
    0.551480109396776, 0.5681764840098174, 0.5221898626065292, 
    0.5549640590370928, 0.561192037936224, 0.5387187718648393, 
    0.5212856906822424, 0.5453574794000325, 0.5182388148680284, 
    0.558590763241926, 0.5432499318862956, 0.5404501786369919, 
    0.5208571547054693, 0.5715308544112891, 0.5456000072279142,
    0.5806533667797404, 0.5620355107855107, 0.539954320370987,
    0.5480145625978959, 0.547126843194259, 0.54536386908546, 
    0.5576249981079526, 0.5474022736522737, 0.5567650405150405, 
    0.5608146698262977
])

epoch = np.arange(1, 41)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.plot(epoch, training_loss_exp_2, label='Training Loss')
ax1.plot(epoch, evaluation_loss_exp_2, label='Evaluation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epoch, training_acc_exp_2, label='Training Accuracy')
ax2.plot(epoch, evaluation_acc_exp_2, label='Evaluation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()