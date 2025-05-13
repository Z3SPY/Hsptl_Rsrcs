from hospital_env import HospitalSimEnv
import numpy as np, random, pprint, warnings
warnings.filterwarnings("ignore")

env = HospitalSimEnv({})
obs = env.reset()
total = 0
for t in range(20):           # ~ one simulated week with 8-h shifts
    a = np.zeros(8, np.float32)
    a[:6] = np.random.uniform(-1, 1, 6)
    a[6]  = 1.0               # aggressive discharge
    a[7]  = 0.7               # 70% deferral
    obs, r, done, info = env.step(a)
    total += r
    print(f"Shift {t:02d}  r={r:5.2f}  ICUq={obs[-2]*10:.0f}")
print("Total reward:", total)
