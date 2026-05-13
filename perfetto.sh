adb shell perfetto \
-o /data/misc/perfetto-traces/systrace.pftrace \
-t 15s \
sched freq idle memory disk workq power binder_driver

adb shell simpleperf stat -p <PID> -e l1d_cache_refill,l2d_cache_refill,bus_access --duration 5
