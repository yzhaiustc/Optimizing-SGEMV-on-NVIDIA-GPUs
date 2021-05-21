# Simple-SGEMV-on-GPU
An implementation of SGEMV with performance comparable to cuBLAS.

## Sample run on Nvidia RTX 2080 Super.

Sample run1 (testing ```mysgemv```):
```
./sgemv 20480 20480 1 
m = 20480, n = 20480.
Testing my sgemv.
Start the sanity check...
Sanity check passed. Start performance benchmarking...
Average elasped time: 0.003594 second, performance: 233.399858 GFLOPS.
```

Sample run2 (testing ```cublasSgemv```):
```
./sgemv 20480 20480 2
m = 20480, n = 20480.
Testing cuBLAS SGEMV.
Average elasped time: 0.003627 second, performance: 231.293983 GFLOPS.
```

Full data can be found [here](https://github.com/yzhaiustc/Simple-SGEMV-on-GPU/tree/main/data).
