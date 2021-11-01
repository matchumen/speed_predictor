import numpy as np
from utils import MAE


def evaluate_chunks(data):
    chunk_len = 100
    chunks_results = []
    for td in data['drives_results']:
        for i in range(0, td['measured'].shape[0], chunk_len):
            mean = np.mean(td['measured'][i:i+chunk_len])
            mean_predicted = np.mean(td['predicted'][i:i+chunk_len])
            std = np.std(td['measured'][i:i+chunk_len])
            mae = MAE(td['measured'][i:i+chunk_len], td['predicted'][i:i+chunk_len])
            mbd = MAE(td['measured'][i:i+chunk_len], td['baseline'][i:i+chunk_len]) - mae
            chunk_result = {
                'source': td['pth'],
                'mean': mean,
                'std': std,
                'MAE': mae,
                'MBD': mbd,
                'mean_predicted': mean_predicted
            }

            chunks_results.append(chunk_result)
            
    data['chunks_results'] = chunks_results
    
    return data


def SDMAE(y, ycap):
    window_size = 50
    
    y = np.convolve(y, np.ones(window_size), 'valid') / window_size
    
    ycap = np.convolve(ycap, np.ones(window_size), 'valid') / window_size
    
    return MAE(np.diff(y), np.diff(ycap))


def evaluate_results(drives_results):
    for tr in drives_results:
        tr['MAE'] = MAE(tr['measured'], tr['predicted'])
        tr['MBD'] = MAE(tr['measured'], tr['baseline']) - tr['MAE']
        tr['SDMAE'] = SDMAE(tr['measured'], tr['predicted'])
        
    overall_results = {
        'drives_results': drives_results,
        'mean_MAE': np.mean(np.array([dr['MAE'] for dr in drives_results])),
        'median_MAE': np.median(np.array([dr['MAE'] for dr in drives_results])),
        'mean_MBD': np.mean(np.array([dr['MBD'] for dr in drives_results])),
        'median_MBD': np.median(np.array([dr['MBD'] for dr in drives_results])),
        'improved_ratio': len([tr['MBD'] for tr in drives_results if tr['MBD'] > 0]) / len(drives_results)
    }
    
    overall_results = evaluate_chunks(overall_results)

    overall_results = evaluate_cars(overall_results)

    overall_results = cluster_chunks(overall_results)

    overall_results = cluster_chunks_std(overall_results)
    
    return overall_results


def evaluate_cars(res):
    car_results = {}
        
    for d in res['drives_results']:
        car_id = d['pth'].split('-')[1]
        if car_id not in car_results.keys():
            car_results[car_id] = {}
            car_results[car_id]['MAE'] = []
            car_results[car_id]['MBD'] = []
        car_results[car_id]['MAE'].append(d['MAE'])
        car_results[car_id]['MBD'].append(d['MBD'])
        
    for car in car_results.keys():
        car_results[car]['mean_MAE'] = np.mean(car_results[car]['MAE'])
        car_results[car]['median_MAE'] = np.median(car_results[car]['MAE'])
        car_results[car]['mean_MBD'] = np.mean(car_results[car]['MBD'])
        car_results[car]['median_MBD'] = np.median(car_results[car]['MBD'])
        car_results[car].pop('MAE')
        car_results[car].pop('MBD')
        
    res['car_results'] = car_results
    
    return res


def cluster_chunks(res, bins=15):
    bin_size = 60/bins
    etalons = np.arange(bin_size//2, bin_size * bins, bin_size)

    cluster_results = {}

    for c in res['chunks_results']:
        bin = str(etalons[np.argmin(np.abs(etalons - c['mean']))])
        if bin not in cluster_results.keys():
            cluster_results[bin] = {'MAE': []}

        cluster_results[bin]['MAE'].append(c['MAE'])

    for key, _ in cluster_results.items():
        cluster_results[key]['mean_MAE'] = np.mean(cluster_results[key]['MAE'])
        cluster_results[key]['median_MAE'] = np.median(cluster_results[key]['MAE'])

    res['cluster_results'] = cluster_results

    return res


def cluster_chunks_std(res, bins=15):
    bin_size = 10/bins
    etalons = np.arange(bin_size//2, bin_size * bins, bin_size)

    cluster_results = {}

    for c in res['chunks_results']:
        bin = str(etalons[np.argmin(np.abs(etalons - c['std']))])
        if bin not in cluster_results.keys():
            cluster_results[bin] = {'MAE': []}

        cluster_results[bin]['MAE'].append(c['MAE'])

    for key, _ in cluster_results.items():
        cluster_results[key]['mean_MAE'] = np.mean(cluster_results[key]['MAE'])
        cluster_results[key]['median_MAE'] = np.median(cluster_results[key]['MAE'])

    res['cluster_results_std'] = cluster_results

    return res
