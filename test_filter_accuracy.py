from scipy.signal import lfilter_zi
from scipy.signal import butter
from scipy.signal._signaltools import (
    lfilter_zi_main_impl,
    lfilter_zi_mpmath,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
#np.set_printoptions(precision=16)
from mpmath import mp
import time
mp.dps = 200


b = [1., 0., 0.]
a = [1., 1., -6.]
x = [8., 8., 8., 8., 8., 8., 8., 8.]

#out1 = lfilter_zi(b, a)
#print(out1)
#out2 = lfilter_zi_main_impl(b, a)
#print(out2)
#out3 = lfilter_zi_mpmath(b, a)
#print(out3)
#breakpoint()
def find_error(actual, desired):
    assert actual.ndim == 1
    assert desired.ndim == 1
    error_abs = np.linalg.norm(actual - desired)
    error_rel = error_abs / np.linalg.norm(desired)
    return error_rel

def gen_butterworth():
    btype = np.random.choice(['lowpass', 'highpass'])
    N = np.random.randint(4, 10)
    Wn = np.random.uniform(0.1, 0.9)
    #print(N, Wn, btype)
    #b, a = butter(np.random.randint(5, 10), np.random.uniform(100, 1000), output='ba')
    b, a = butter(N, Wn, output='ba', btype=btype)
    return b, a

def gen_random():
    b = np.random.normal(size=20)
    a = np.random.normal(size=20)
    return b, a

def test_lfilter_accuracy(generate_func, N):
    old_err = []
    new_err = []
    inputs = []
    start = time.time()
    for i in tqdm(range(N)):
        #a = np.random.normal(size=10)
        #b = np.random.normal(size=10)
        #coeff_scale = np.arange(1, 11)
        #coeff_scale = np.ones(10)
        #a = a * coeff_scale
        #b = b * coeff_scale
        b, a = generate_func()
        #print(b, a)
        if a[0] == 0:
            print("avoiding div by zero")
            continue
        ref_zi = lfilter_zi_mpmath(b, a)
        new_zi = lfilter_zi(b, a)
        old_zi = lfilter_zi_main_impl(b, a)
        #print()
        #print(i)
        #print(ref_zi)
        #print(new_zi - ref_zi)
        #print(old_zi - ref_zi)
        #print("new error", find_error(new_zi, ref_zi))
        #print("old error", find_error(old_zi, ref_zi))
        old_err.append(find_error(old_zi, ref_zi))
        new_err.append(find_error(new_zi, ref_zi))
        inputs.append({"a": a, "b": b})
        #print("old", old_err[-1])
        #print("new", new_err[-1])
        #print()
    df = pd.DataFrame({"old_err": old_err, "new_err": new_err, "inputs": inputs})
    df[["old_err", "new_err"]] /= np.finfo('float64').eps
    return df

def display_error(df):
    print("Error measurement, in ULPs")
    print(df[["old_err", "new_err"]].describe())
    #print("worst old")
    #print(df.loc[df["old_err"].idxmax()]["inputs"])
    #print("worst new")
    #print(df.loc[df["new_err"].idxmax()]["inputs"])
    print("% of cases where old_err > new_err")
    print((df["old_err"] > df["new_err"]).mean() * 100)


def main():
    print("random")
    df = test_lfilter_accuracy(gen_random, 5000)
    display_error(df)

    print("butterworth")
    df = test_lfilter_accuracy(gen_butterworth, 50000)
    display_error(df)

if __name__ == "__main__":
    main()
