# Cache-Aware Structured Pruning — Research Run Summary

- Run mode: **college_v1_30e** (smoke test=False)
- Architecture: DenseNet-BC-100-12; dataset: CIFAR-10
- Device: cuda; GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Precision: AMP; seed: 42
- Cache reference: 1.500 MiB (reported by PyTorch device property L2_cache_size)
- Cache working-set estimates are proxies; no hardware L2 hit/miss counter result is claimed.
- Checkpoint file size refers to serialized model-only inference weights; separate best/latest training checkpoints preserve optimizer/scheduler state for recovery.
- NVML power is total device power. Integrated energy is the trapezoidal integral of sampled power; incremental energy subtracts an idle estimate when available.

## Three-way measured comparison

                          model  test_loss  test_accuracy_percent  parameters  macs_per_image  checkpoint_file_size_mb  total_training_time_sec  latency_image_ms  latency_batch_median_ms  throughput_images_per_sec  peak_gpu_memory_allocated_mb  peak_gpu_memory_reserved_mb  avg_gpu_power_w  energy_per_image_mj  parameter_reduction_percent  mac_reduction_percent
    Baseline DenseNet-BC-100-12   0.227404                  92.96      769162       287929692                 3.249766              8860.125573          0.823218                105.29310                1214.744377                    486.516602                       3156.0        61.267203            50.991272                     0.000000               0.000000
Conventional structured pruning   0.240627                  92.52      462986       162690800                 2.071084               547.962064          0.644149                 82.48035                1552.435950                    425.926270                       2542.0        51.719390            34.171820                    39.806439              43.496345
 Cache-aware structured pruning   0.234192                  92.64      455130       161208792                 2.039382               526.490241          1.013278                 95.71225                 986.896127                    381.204102                       2376.0        56.382903            36.173335                    40.827810              44.011057

## Interpretation guardrails

- Difference in achieved MAC reduction between conventional and cache-aware pruning: **0.515 percentage points**.
- Matching tolerance configured: 5.0 percentage points.
- Report only the measurements above. A smoke-test run is for pipeline validation and must not be treated as a final accuracy benchmark.
- Latency and energy depend on hardware, runtime load, precision, and measurement conditions; repeat measurements for a research claim.
- FLOPs are estimated as 2 × Conv/Linear MACs; non-Conv/Linear operations are not included in this estimate.
