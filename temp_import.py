import traceback
try:
    import scipy.stats
except Exception as e:
    with open("err2.txt", "w") as f:
        f.write(traceback.format_exc())
