adb shell perfetto \
-o /data/misc/perfetto-traces/systrace.pftrace \
-t 15s \
sched freq idle memory disk workq power binder_driver
