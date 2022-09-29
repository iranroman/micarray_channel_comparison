import fire
import librosa
import itertools
import numpy as np
import tqdm
from sklearn import metrics
from pathlib import Path
from scipy.fft import fft

criteria = {
    'mse': metrics.mean_squared_error,

}

def get_all_wav_files(datadir, criterion, features):
    path = Path(datadir)
    wav_files = [*path.glob('**/*.wav')]

    criterion = getattr(metrics, criterion)
    
    # per channel pairs across the 
    # entire dataset
    all_distances = {}
    for w in tqdm.tqdm(wav_files[:10]):
        x, sr = librosa.load(w,mono=False, sr=None)
        nchans = x.shape[0]
        pairs = list(itertools.combinations(range(nchans),2))
        for p in pairs:
            if p not in all_distances:
                all_distances[p] = []
            if features == 'raw':
                all_distances[p].append(criterion(x[p[0]],x[p[1]]))
            elif features == 'stft':
                a = librosa.stft(x[p[0]])
                b = librosa.stft(x[p[1]])
                all_distances[p].append(criterion(np.log(np.abs(a)+1e-20),np.log(np.abs(b)+1e-20)))
            elif features == 'fft':
                a = fft(x[p[0]])
                a /= len(a)
                a = np.concatenate((np.real(a),np.imag(a)))
                b = fft(x[p[1]])
                b /= len(b)
                b = np.concatenate((np.real(b),np.imag(b)))
                all_distances[p].append(criterion(a,b))
    print([(k,np.mean(v)) for k,v in all_distances.items()])
            

def main(
        datadir = None,
        criterion = None,
        features = None,
        ):

    fnames = get_all_wav_files(datadir, criterion, features)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
