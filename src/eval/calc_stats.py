import numpy as np
import sys
import numpy as np

errors = []
for error_file in sys.argv[1:]:
  error = np.loadtxt(error_file)
  print(error_file, error.mean(0))
  errors.append(error)
errors = np.concatenate(errors, axis=0)
print('correspondence errors) before reg: {}, after reg: {}'.format(errors.mean(0)[0], errors.mean(0)[1]))
mask1 = (errors < 0.05).astype(np.float32).mean(0)
print('recall within 5cm)     before reg: {}, after reg: {}'.format(mask1[0], mask1[1]))
mask2 = (errors < 0.1).astype(np.float32).mean(0)
print('recall within 10cm)    before reg: {}, after reg: {}'.format(mask2[0], mask2[1]))
#print((errors < 0.1).astype(np.float32).mean(0))
