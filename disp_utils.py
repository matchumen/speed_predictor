import matplotlib.pyplot as plt
import numpy as np
import utils
import pickle


def print_res(results):
    print("Overall results:")
    print(f"Mean MAE:          {results['mean_MAE']:.3f} mps")
    print(f"Median MAE:        {results['median_MAE']:.3f} mps")
    print(f"Mean MBD:          {results['mean_MBD']:.3f} mps")
    print(f"Median MBD:        {results['median_MBD']:.3f} mps")
    print(f"Improvement ratio: {results['improved_ratio']*100:.1f}%")

    print("")
    print("Carwise results:")
    for k, results in results['car_results'].items():
        print(k)
        print(f"Mean MAE:          {results['mean_MAE']:.3f} mps")
        print(f"Median MAE:        {results['median_MAE']:.3f} mps")
        print(f"Mean MBD:          {results['mean_MBD']:.3f} mps")
        print(f"Median MBD:        {results['median_MBD']:.3f} mps")

    print("")
    print("Speedwise results")


def mae_std_clustered(test_results):
    y = [test_results['cluster_results_std'][key]['mean_MAE'] for key in test_results['cluster_results_std'].keys()]
    x = [float(k) for k in test_results['cluster_results_std'].keys()]

    plt.figure(figsize=(10, 4))
    plt.title('Rozdělení absolutní chyby podle výběrové směrodatné odchylky')
    plt.xlabel('Výběrová střední odchylka [m/s]')
    plt.ylabel('Absolutní chyba modelu [m/s]')
    plt.bar(x, y, tick_label=x)
    plt.xticks(rotation=45, ha="right")

    plt.show()


def mae_cars(test_results):
    x = list(test_results['car_results'].keys())
    y = [test_results['car_results'][key]['mean_MAE'] for key in test_results['car_results'].keys()]

    plt.figure(figsize=(10, 4))
    plt.title('Rozdělení absolutní chyby podle vozidel')
    plt.xlabel('ID vozidla')
    plt.ylabel('Absolutní chyba modelu [m/s]')
    plt.bar(x, y, tick_label=x)
    plt.show()


def mae_mean_clustered(test_results):
    y = [test_results['cluster_results'][key]['mean_MAE'] for key in test_results['cluster_results'].keys()]
    x = [float(k) for k in test_results['cluster_results'].keys()]

    plt.figure(figsize=(10, 4))
    plt.title('Rozdělení absolutní chyby podle rychlosti')
    plt.xlabel('Naměřená rychlost [m/s]')
    plt.ylabel('Průměrná bsolutní chyba modelu [m/s]')
    plt.bar(x, y, tick_label=x)
    plt.show()


def model_to_measured(test_results):
    chunks_results = test_results['chunks_results']
    mean = [cr['mean'] for cr in chunks_results]
    mae = [cr['MAE'] for cr in chunks_results]
    mean_pred = [cr['mean_predicted'] for cr in chunks_results]
    std = [cr['std'] for cr in chunks_results]
    source = [cr['source'] for cr in chunks_results]
    source = [s.split('-')[1] for s in source]
    cmap = {
        '002': 'red',
        '004': 'yellow',
        '005': 'green',
        '006': 'blue'
    }
    source = [cmap[s] for s in source]
    plt.figure(figsize=(10, 10))
    plt.title('Závislost naměřené a predikované rychlosti')
    plt.xlabel('Průměrná naměřená rychlost [m/s]')
    plt.ylabel('Průměrná predikovaná rychlost [m/s]')
    #plt.hist2d(mean, mean_pred, bins=(200, 200), cmap=plt.cm.jet)
    #plt.vlines(np.arange(60), 0, 60)
    #plt.hlines(np.arange(60), 0, 60)
    plt.grid()
    plt.scatter(mean, mean_pred, s=1, c=source, alpha=0.5, label=list(cmap.keys()))
    plt.plot([0, 50], [0, 50], c='red', label='', alpha=0.8)
    plt.show()


def speed_profile(test_results, drive_id):
    fu = test_results['drives_results'][drive_id]
    plt.figure(figsize=(20, 10))
    plt.title('Srovnání naměřeného a predikovaného rychlostního profilu')
    plt.plot(fu['measured'], label='naměřený')
    plt.plot(fu['predicted'], label='predikovaný')
    #plt.plot(fu['baseline'], label='baseline')
    plt.xlabel('vzdálenost [m]')
    plt.ylabel('rychost[m/s]')
    plt.legend(loc='upper right')
    plt.show()


def plot_study_history(history, k_best=None):
    if k_best:
        cutoff = k_best
    else:
        cutoff = len(history)
    
    plt.figure(figsize=(20, 10))
    plt.yticks(np.arange(10) * 0.1)
    plt.xticks(np.arange(len(history[0][1][0])))
    plt.grid()
    for h in history[:cutoff]:
        plt.plot(np.arange(len(h[1][0][1:])), h[1][0][1:], '--', label=f'trial {h[0]:.6f} train')
        plt.plot(np.arange(len(h[1][1][1:])), h[1][1][1:], label=f'trial {h[0]:.6f} val')
        plt.legend()
    plt.show()


def load_study_history(study_name, study_dir, storage):
    trials = utils.load_trials(study_name, storage)
    vals = [t.value for t in trials]
    hists = []
    for val in vals:
        with open(f'{study_dir}/history/history_{val:.6f}.pickle', 'rb') as f:
            hist = pickle.load(f)
            hists.append((val, hist))
    return hists